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
import os
import six

import chainer
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L

# for classifier link
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter
from chainer import training
from chainer.training import extensions

from espnet.lm.lm_utils import compute_perplexity
from espnet.lm.lm_utils import count_tokens
from espnet.lm.lm_utils import MakeSymlinkToBestModel
from espnet.lm.lm_utils import ParallelSentenceIterator
from espnet.lm.lm_utils import read_tokens

import espnet.nets.chainer_backend.deterministic_embed_id as DL

REPORT_INTERVAL = 100


class ClassifierWithState(link.Chain):
    """A wrapper for a chainer RNNLM

    :param link.Chain predictor : The RNNLM
    :param function lossfun: The loss function to use
    :param int/str label_key:
    """

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 label_key=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_key))

        super(ClassifierWithState, self).__init__()
        self.lossfun = lossfun
        self.y = None
        self.loss = None
        self.label_key = label_key

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, state, *args, **kwargs):
        """Computes the loss value for an input and label pair.

            It also computes accuracy and stores it to the attribute.
            When ``label_key`` is ``int``, the corresponding element in ``args``
            is treated as ground truth labels. And when it is ``str``, the
            element in ``kwargs`` is used.
            The all elements of ``args`` and ``kwargs`` except the groundtruth
            labels are features.
            It feeds features to the predictor and compare the result
            with ground truth labels.

        :param state : The LM state
        :param list[chainer.Variable] args : Input minibatch
        :param dict[chainer.Variable] kwargs : Input minibatch
        :return loss value
        :rtype chainer.Variable
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

        :param state : the state
        :param x : the input
        :return a tuple (state, log prob vector)
        :rtype cupy/numpy array
        """
        if hasattr(self.predictor, 'normalized') and self.predictor.normalized:
            return self.predictor(state, x)
        else:
            state, z = self.predictor(state, x)
            return state, F.log_softmax(z).data

    def final(self, state):
        """Predict final log probabilities for given state using the predictor

        :param state : the state
        :return log probability vector
        :rtype cupy/numpy array

        """
        if hasattr(self.predictor, 'final'):
            return self.predictor.final(state)
        else:
            return 0.


# Definition of a recurrent net for language modeling
class RNNLM(chainer.Chain):
    """A chainer RNNLM

    :param int n_vocab: The size of the vocabulary
    :param int n_layers: The number of layers to create
    :param int n_units: The number of units per layer
    """

    def __init__(self, n_vocab, n_layers, n_units):
        super(RNNLM, self).__init__()
        with self.init_scope():
            self.embed = DL.EmbedID(n_vocab, n_units)
            self.lstm = chainer.ChainList(
                *[L.StatelessLSTM(n_units, n_units) for _ in range(n_layers)])
            self.lo = L.Linear(n_units, n_vocab)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.n_layers = n_layers

    def __call__(self, state, x):
        if state is None:
            state = {'c': [None] * self.n_layers, 'h': [None] * self.n_layers}
        h = [None] * self.n_layers
        c = [None] * self.n_layers
        emb = self.embed(x)
        c[0], h[0] = self.lstm[0](state['c'][0], state['h'][0], F.dropout(emb))
        for n in six.moves.range(1, self.n_layers):
            c[n], h[n] = self.lstm[n](state['c'][n], state['h'][n], F.dropout(h[n - 1]))
        y = self.lo(F.dropout(h[-1]))
        state = {'c': c, 'h': h}
        return state, y


class BPTTUpdater(training.updaters.StandardUpdater):
    """An updater for a chainer LM

    :param chainer.dataset.Iterator train_iter : The train iterator
    :param optimizer:
    :param int device : The device id
    """

    def __init__(self, train_iter, optimizer, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        # Progress the dataset iterator for sentences at each iteration.
        batch = train_iter.__next__()
        x, t = convert.concat_examples(batch, device=self.device, padding=(0, -1))
        # Concatenate the token IDs to matrices and send them to the device
        # self.converter does this job
        # (it is chainer.dataset.concat_examples by default)
        xp = chainer.backends.cuda.get_array_module(x)
        loss = 0
        count = 0
        state = None
        batch_size, sequence_length = x.shape
        for i in six.moves.range(sequence_length):
            # Compute the loss at this time step and accumulate it
            state, loss_batch = optimizer.target(state, chainer.Variable(x[:, i]),
                                                 chainer.Variable(t[:, i]))
            non_zeros = xp.count_nonzero(x[:, i])
            loss += loss_batch * non_zeros
            count += int(non_zeros)

        reporter.report({'loss': float(loss.data)}, optimizer.target)
        reporter.report({'count': count}, optimizer.target)
        # update
        loss /= batch_size  # normalized by batch size
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters


class LMEvaluator(extensions.Evaluator):
    """A custom evaluator for a chainer LM

    :param chainer.dataset.Iterator val_iter : The validation iterator
    :param eval_model : The model to evaluate
    :param int device : The device id to use
    """

    def __init__(self, val_iter, eval_model, device):
        super(LMEvaluator, self).__init__(
            val_iter, eval_model, device=device)

    def evaluate(self):
        val_iter = self.get_iterator('main')
        target = self.get_target('main')
        loss = 0
        count = 0
        for batch in copy.copy(val_iter):
            x, t = convert.concat_examples(batch, device=self.device, padding=(0, -1))
            xp = chainer.backends.cuda.get_array_module(x)
            state = None
            for i in six.moves.range(len(x[0])):
                state, loss_batch = target(state, x[:, i], t[:, i])
                non_zeros = xp.count_nonzero(x[:, i])
                loss += loss_batch.data * non_zeros
                count += int(non_zeros)
        # report validation loss
        observation = {}
        with reporter.report_scope(observation):
            reporter.report({'loss': float(loss / count)}, target)
        return observation


def train(args):
    """Train with the given args

    :param namespace args: The program arguments
    """
    # display chainer version
    logging.info('chainer version = ' + chainer.__version__)

    # seed setting (chainer seed may not need it)
    nseed = args.seed
    os.environ['CHAINER_SEED'] = str(nseed)
    logging.info('chainer seed = ' + os.environ['CHAINER_SEED'])

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # remove type check
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info('chainer type check is disabled')
    # use deterministic computation or not
    if args.debugmode < 1:
        chainer.config.cudnn_deterministic = False
        logging.info('chainer cudnn deterministic is disabled')
    else:
        chainer.config.cudnn_deterministic = True

    # check cuda and cudnn availability
    if not chainer.cuda.available:
        logging.warning('cuda is not available')
    if not chainer.cuda.cudnn_enabled:
        logging.warning('cudnn is not available')

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
    rnn = RNNLM(args.n_vocab, args.layer, args.unit)
    model = ClassifierWithState(rnn)
    if args.ngpu > 1:
        logging.warning("currently, multi-gpu is not supported. use single gpu.")
    if args.ngpu > 0:
        # Make the specified GPU current
        gpu_id = 0
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    else:
        gpu_id = -1

    # Save model conf to json
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps(vars(args), indent=4, sort_keys=True).encode('utf_8'))

    # Set up an optimizer
    if args.opt == 'sgd':
        optimizer = chainer.optimizers.SGD(lr=1.0)
    elif args.opt == 'adam':
        optimizer = chainer.optimizers.Adam()

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    updater = BPTTUpdater(train_iter, optimizer, gpu_id)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)
    trainer.extend(LMEvaluator(val_iter, model, device=gpu_id))
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(REPORT_INTERVAL, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity', 'elapsed_time']
    ), trigger=(REPORT_INTERVAL, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))
    trainer.extend(extensions.snapshot(filename='snapshot.ep.{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(
        model, 'rnnlm.model.{.updater.epoch}'))
    # MEMO(Hori): wants to use MinValueTrigger, but it seems to fail in resuming
    trainer.extend(MakeSymlinkToBestModel('validation/main/loss', 'rnnlm.model'))

    if args.resume:
        logging.info('resumed from %s' % args.resume)
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    # compute perplexity for test set
    if args.test_label:
        logging.info('test the best model')
        chainer.serializers.load_npz(args.outdir + '/rnnlm.model.best', model)
        test = read_tokens(args.test_label, args.char_list_dict)
        n_test_tokens, n_test_oovs = count_tokens(test, unk)
        logging.info('#sentences in the test data = ' + str(len(test)))
        logging.info('#tokens in the test data = ' + str(n_test_tokens))
        logging.info('oov rate in the test data = %.2f %%' % (n_test_oovs / n_test_tokens * 100))
        test_iter = ParallelSentenceIterator(test, args.batchsize,
                                             max_length=args.maxlen, sos=eos, eos=eos, repeat=False)
        evaluator = LMEvaluator(test_iter, model, device=gpu_id)
        with chainer.using_config('train', False):
            result = evaluator()
        logging.info('test perplexity: ' + str(np.exp(float(result['main/loss']))))
