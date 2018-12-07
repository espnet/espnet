#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py

from __future__ import division
from __future__ import print_function

import copy
import json
import logging
import numpy as np
import six

import torch
import torch.nn as nn
import torch.nn.functional as F

from chainer import Chain
from chainer.dataset import convert
from chainer import reporter
from chainer import training
from chainer.training import extensions

from espnet.lm.lm_utils import compute_perplexity
from espnet.lm.lm_utils import count_tokens
from espnet.lm.lm_utils import MakeSymlinkToBestModel
from espnet.lm.lm_utils import ParallelSentenceIterator
from espnet.lm.lm_utils import read_tokens
from espnet.nets.pytorch_backend.e2e_asr import to_device

from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_save
from espnet.asr.asr_utils import torch_snapshot

REPORT_INTERVAL = 100


# dummy module to use chainer's trainer
class Reporter(Chain):
    def report(self, loss):
        pass


class ClassifierWithState(nn.Module):
    """A wrapper for pytorch RNNLM

    :param torch.nn.Module predictor : The RNNLM
    :param function lossfun : The loss function to use
    :param int/str label_key :
    """

    def __init__(self, predictor,
                 lossfun=F.cross_entropy,
                 label_key=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_key))
        super(ClassifierWithState, self).__init__()
        self.lossfun = lossfun
        self.y = None
        self.loss = None
        self.label_key = label_key
        self.predictor = predictor
        self.reporter = Reporter()

    def forward(self, state, *args, **kwargs):
        """Computes the loss value for an input and label pair.az

            It also computes accuracy and stores it to the attribute.
            When ``label_key`` is ``int``, the corresponding element in ``args``
            is treated as ground truth labels. And when it is ``str``, the
            element in ``kwargs`` is used.
            The all elements of ``args`` and ``kwargs`` except the groundtruth
            labels are features.
            It feeds features to the predictor and compare the result
            with ground truth labels.

        :param torch.Tensor state : the LM state
        :param list[torch.Tensor] args : Input minibatch
        :param dict[torch.Tensor] kwargs : Input minibatch
        :return loss value
        :rtype torch.Tensor
        """

        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        state, self.y = self.predictor(state, *args, **kwargs)
        self.loss = self.lossfun(self.y, t)
        return state, self.loss

    def predict(self, state, x):
        """Predict log probabilities for given state and input x using the predictor

        :param torch.Tensor state : The current state
        :param torch.Tensor x : The input
        :return a tuple (new state, log prob vector)
        :rtype (torch.Tensor, torch.Tensor)
        """
        if hasattr(self.predictor, 'normalized') and self.predictor.normalized:
            return self.predictor(state, x)
        else:
            state, z = self.predictor(state, x)
            return state, F.log_softmax(z, dim=1)

    def buff_predict(self, state, x, n):
        if self.predictor.__class__.__name__ == 'RNNLM':
            return self.predict(state, x)

        new_state = []
        new_log_y = []
        for i in range(n):
            state_i = None if state is None else state[i]
            state_i, log_y = self.predict(state_i, x[i].unsqueeze(0))
            new_state.append(state_i)
            new_log_y.append(log_y)

        return new_state, torch.cat(new_log_y)

    def final(self, state):
        """Predict final log probabilities for given state using the predictor

        :param state: The state
        :return The final log probabilities
        :rtype torch.Tensor
        """
        if hasattr(self.predictor, 'final'):
            return self.predictor.final(state)
        else:
            return 0.


# Definition of a recurrent net for language modeling
class RNNLM(nn.Module):
    """A pytorch RNNLM

    :param int n_vocab: The size of the vocabulary
    :param int n_layers: The number of layers to create
    :param int n_units: The number of units per layer
    :param str typ: The RNN type
    """

    def __init__(self, n_vocab, n_layers, n_units, typ="lstm"):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(n_vocab, n_units)
        self.rnn = nn.ModuleList(
            [nn.LSTMCell(n_units, n_units) for _ in range(n_layers)] if typ == "lstm" else [nn.GRUCell(n_units, n_units)
                                                                                            for _ in range(n_layers)])
        self.dropout = nn.ModuleList(
            [nn.Dropout() for _ in range(n_layers + 1)])
        self.lo = nn.Linear(n_units, n_vocab)
        self.n_layers = n_layers
        self.n_units = n_units
        self.typ = typ

        # initialize parameters from uniform distribution
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)

    def zero_state(self, batchsize):
        return torch.zeros(batchsize, self.n_units).float()

    def forward(self, state, x):
        if state is None:
            h = [to_device(self, self.zero_state(x.size(0))) for n in six.moves.range(self.n_layers)]
            state = {'h': h}
            if self.typ == "lstm":
                c = [to_device(self, self.zero_state(x.size(0))) for n in six.moves.range(self.n_layers)]
                state = {'c': c, 'h': h}

        h = [None] * self.n_layers
        emb = self.embed(x)
        if self.typ == "lstm":
            c = [None] * self.n_layers
            h[0], c[0] = self.rnn[0](self.dropout[0](emb), (state['h'][0], state['c'][0]))
            for n in six.moves.range(1, self.n_layers):
                h[n], c[n] = self.rnn[n](self.dropout[n](h[n - 1]), (state['h'][n], state['c'][n]))
            state = {'c': c, 'h': h}
        else:
            h[0] = self.rnn[0](self.dropout[0](emb), state['h'][0])
            for n in six.moves.range(1, self.n_layers):
                h[n] = self.rnn[n](self.dropout[n](h[n - 1]), state['h'][n])
            state = {'h': h}
        y = self.lo(self.dropout[-1](h[-1]))
        return state, y


def concat_examples(batch, device=None, padding=None):
    """Custom concat_examples for pytorch

    :param np.ndarray batch: The batch to concatenate
    :param int device: The device to send to
    :param Tuple[int,int] padding: The padding to use
    :return: (inputs, targets)
    :rtype (torch.Tensor, torch.Tensor)
    """
    x, t = convert.concat_examples(batch, padding=padding)
    x = torch.from_numpy(x)
    t = torch.from_numpy(t)
    if device is not None and device >= 0:
        x = x.cuda(device)
        t = t.cuda(device)
    return x, t


class BPTTUpdater(training.StandardUpdater):
    """An updater for a pytorch LM

    :param chainer.dataset.Iterator train_iter : The train iterator
    :param torch.nn.Module model : The model to update
    :param optimizer:
    :param int device : The device id
    :param int gradclip : The gradient clipping value to use
    """

    def __init__(self, train_iter, model, optimizer, device, gradclip=None):
        super(BPTTUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.device = device
        self.gradclip = gradclip

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        # Progress the dataset iterator for sentences at each iteration.
        batch = train_iter.__next__()
        x, t = concat_examples(batch, device=self.device, padding=(0, -100))
        # Concatenate the token IDs to matrices and send them to the device
        # self.converter does this job
        # (it is chainer.dataset.concat_examples by default)
        loss = 0
        count = 0
        state = None
        batch_size, sequence_length = x.shape
        for i in six.moves.range(sequence_length):
            # Compute the loss at this time step and accumulate it
            state, loss_batch = self.model(state, x[:, i], t[:, i])
            non_zeros = torch.sum(x[:, i] != 0, dtype=torch.float)
            loss += loss_batch * non_zeros
            count += int(non_zeros)

        reporter.report({'loss': float(loss.detach())}, optimizer.target)
        reporter.report({'count': count}, optimizer.target)
        # update
        loss = loss / batch_size  # normalized by batch size
        self.model.zero_grad()  # Clear the parameter gradients
        loss.backward()  # Backprop
        if self.gradclip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradclip)
        optimizer.step()  # Update the parameters


class LMEvaluator(extensions.Evaluator):
    """A custom evaluator for a pytorch LM

    :param chainer.dataset.Iterator val_iter : The validation iterator
    :param torch.nn.Module eval_model : The model to evaluate
    :param chainer.Reporter reporter : The observations reporter
    :param int device : The device id to use
    """

    def __init__(self, val_iter, eval_model, reporter, device):
        super(LMEvaluator, self).__init__(
            val_iter, reporter, device=device)
        self.model = eval_model

    def evaluate(self):
        val_iter = self.get_iterator('main')
        loss = 0
        count = 0
        self.model.eval()
        with torch.no_grad():
            for batch in copy.copy(val_iter):
                x, t = concat_examples(batch, device=self.device, padding=(0, -100))
                state = None
                for i in six.moves.range(len(x[0])):
                    state, loss_batch = self.model(state, x[:, i], t[:, i])
                    non_zeros = torch.sum(x[:, i] != 0, dtype=torch.float)
                    loss += loss_batch * non_zeros
                    count += int(non_zeros)
        self.model.train()
        # report validation loss
        observation = {}
        with reporter.report_scope(observation):
            reporter.report({'loss': float(loss / count)}, self.model.reporter)
        return observation


def train(args):
    """Train with the given args

    :param Namespace args: The program arguments
    """
    # display torch version
    logging.info('torch version = ' + torch.__version__)

    # seed setting
    nseed = args.seed
    torch.manual_seed(nseed)
    logging.info('torch seed = ' + str(nseed))

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # use deterministic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        logging.info('torch cudnn deterministic is disabled')
    else:
        torch.backends.cudnn.deterministic = True

    # check cuda and cudnn availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get special label ids
    unk = args.char_list_dict['<unk>']
    eos = args.char_list_dict['<eos>']
    # read tokens as a sequence of sentences
    train = read_tokens(args.train_label, args.char_list_dict)
    val = read_tokens(args.valid_label, args.char_list_dict)
    # count tokens
    n_train_tokens, n_train_oovs = count_tokens(train, unk)
    n_val_tokens, n_val_oovs = count_tokens(val, unk)
    logging.info('#vocab = ' + str(args.n_vocab))
    logging.info('#sentences in the training data = ' + str(len(train)))
    logging.info('#tokens in the training data = ' + str(n_train_tokens))
    logging.info('oov rate in the training data = %.2f %%' % (n_train_oovs / n_train_tokens * 100))
    logging.info('#sentences in the validation data = ' + str(len(val)))
    logging.info('#tokens in the validation data = ' + str(n_val_tokens))
    logging.info('oov rate in the validation data = %.2f %%' % (n_val_oovs / n_val_tokens * 100))

    # Create the dataset iterators
    train_iter = ParallelSentenceIterator(train, args.batchsize,
                                          max_length=args.maxlen, sos=eos, eos=eos)
    val_iter = ParallelSentenceIterator(val, args.batchsize,
                                        max_length=args.maxlen, sos=eos, eos=eos, repeat=False)
    logging.info('#iterations per epoch = ' + str(len(train_iter.batch_indices)))
    logging.info('#total iterations = ' + str(args.epoch * len(train_iter.batch_indices)))
    # Prepare an RNNLM model
    rnn = RNNLM(args.n_vocab, args.layer, args.unit, args.type)
    model = ClassifierWithState(rnn)
    if args.ngpu > 1:
        logging.warning("currently, multi-gpu is not supported. use single gpu.")
    if args.ngpu > 0:
        # Make the specified GPU current
        gpu_id = 0
        model.cuda(gpu_id)
    else:
        gpu_id = -1

    # Save model conf to json
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps(vars(args), indent=4, sort_keys=True).encode('utf_8'))

    # Set up an optimizer
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters())

    # FIXME: TOO DIRTY HACK
    reporter = model.reporter
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    updater = BPTTUpdater(train_iter, model, optimizer, gpu_id, gradclip=args.gradclip)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)
    trainer.extend(LMEvaluator(val_iter, model, reporter, device=gpu_id))
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(REPORT_INTERVAL, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity', 'elapsed_time']
    ), trigger=(REPORT_INTERVAL, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))
    # Save best models
    trainer.extend(torch_snapshot(filename='snapshot.ep.{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(
        model, 'rnnlm.model.{.updater.epoch}', savefun=torch_save))
    # T.Hori: MinValueTrigger should be used, but it fails when resuming
    trainer.extend(MakeSymlinkToBestModel('validation/main/loss', 'rnnlm.model'))

    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    trainer.run()

    # compute perplexity for test set
    if args.test_label:
        logging.info('test the best model')
        torch_load(args.outdir + '/rnnlm.model.best', model)
        test = read_tokens(args.test_label, args.char_list_dict)
        n_test_tokens, n_test_oovs = count_tokens(test, unk)
        logging.info('#sentences in the test data = ' + str(len(test)))
        logging.info('#tokens in the test data = ' + str(n_test_tokens))
        logging.info('oov rate in the test data = %.2f %%' % (n_test_oovs / n_test_tokens * 100))
        test_iter = ParallelSentenceIterator(test, args.batchsize,
                                             max_length=args.maxlen, sos=eos, eos=eos, repeat=False)
        evaluator = LMEvaluator(test_iter, model, reporter, device=gpu_id)
        result = evaluator()
        logging.info('test perplexity: ' + str(np.exp(float(result['main/loss']))))
