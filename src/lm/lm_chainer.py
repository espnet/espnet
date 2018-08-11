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

import chainer
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L
from chainer import serializers

# for classifier link
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter

from lm_utils import ParallelSequentialIterator


class ClassifierWithState(link.Chain):
    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy,
                 label_key=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_key))

        super(ClassifierWithState, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.label_key = label_key

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, state, *args, **kwargs):
        """Computes the loss value for an input and label pair.

        It also computes accuracy and stores it to the attribute.

        Args:
            args (list of ~chainer.Variable): Input minibatch.
            kwargs (dict of ~chainer.Variable): Input minibatch.

        When ``label_key`` is ``int``, the correpoding element in ``args``
        is treated as ground truth labels. And when it is ``str``, the
        element in ``kwargs`` is used.
        The all elements of ``args`` and ``kwargs`` except the ground trush
        labels are features.
        It feeds features to the predictor and compare the result
        with ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

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
        self.accuracy = None
        state, self.y = self.predictor(state, *args, **kwargs)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return state, self.loss

    def predict(self, state, x):
        """Predict log probabilities for given state and input x using the predictor

        Returns:
            any type: new state
            cupy/numpy array: log probability vector

        """
        if hasattr(self.predictor, 'normalized') and self.predictor.normalized:
            return self.predictor(state, x)
        else:
            state, z = self.predictor(state, x)
            return state, F.log_softmax(z).data


# Definition of a recurrent net for language modeling
class RNNLM(chainer.Chain):

    def __init__(self, n_vocab, n_units):
        super(RNNLM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.l1 = L.StatelessLSTM(n_units, n_units)
            self.l2 = L.StatelessLSTM(n_units, n_units)
            self.lo = L.Linear(n_units, n_vocab)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def __call__(self, state, x):
        if state is None:
            state = {'c1': None, 'h1': None, 'c2': None, 'h2': None}
        h0 = self.embed(x)
        c1, h1 = self.l1(state['c1'], state['h1'], F.dropout(h0))
        c2, h2 = self.l2(state['c2'], state['h2'], F.dropout(h1))
        y = self.lo(F.dropout(h2))
        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
        return state, y


def train(args):
    # display chainer version
    logging.info('chainer version = ' + chainer.__version__)

    # seed setting (chainer seed may not need it)
    nseed = args.seed
    os.environ['CHAINER_SEED'] = str(nseed)
    logging.info('chainer seed = ' + os.environ['CHAINER_SEED'])

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # revmoe type check
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info('chainer type check is disabled')
    # use determinisitic computation or not
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

    with open(args.train_label, 'rb') as f:
        train = np.array([args.char_list_dict[char]
                          if char in args.char_list_dict else args.char_list_dict['<unk>']
                          for char in f.readline().decode('utf-8').split()], dtype=np.int32)
    with open(args.valid_label, 'rb') as f:
        valid = np.array([args.char_list_dict[char]
                          if char in args.char_list_dict else args.char_list_dict['<unk>']
                          for char in f.readline().decode('utf-8').split()], dtype=np.int32)

    logging.info('#vocab = ' + str(args.n_vocab))
    logging.info('#words in the training data = ' + str(len(train)))
    logging.info('#words in the validation data = ' + str(len(valid)))
    logging.info('#iterations per epoch = ' + str(len(train) // (args.batchsize * args.bproplen)))
    logging.info('#total iterations = ' + str(args.epoch * len(train) // (args.batchsize * args.bproplen)))

    # Create the dataset iterators
    train_iter = ParallelSequentialIterator(train, args.batchsize)
    valid_iter = ParallelSequentialIterator(valid, args.batchsize, repeat=False)

    # Prepare an RNNLM model
    rnn = RNNLM(args.n_vocab, args.unit)
    model = ClassifierWithState(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    if args.ngpu > 1:
        logging.warn("currently, multi-gpu is not supported. use single gpu.")
    if args.ngpu > 0:
        # Make the specified GPU current
        gpu_id = 0
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()

    # Save model conf to json
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        f.write(json.dumps(vars(args), indent=4, sort_keys=True).encode('utf_8'))

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    def evaluate(model, iter, bproplen=100):
        # Evaluation routine to be used for validation and test.
        model.predictor.train = False
        evaluator = model.copy()  # to use different state
        state = None
        evaluator.predictor.train = False  # dropout does nothing
        sum_perp = 0
        data_count = 0
        for batch in copy.copy(iter):
            x, t = convert.concat_examples(batch, gpu_id)
            state, loss = evaluator(state, x, t)
            sum_perp += loss.data
            if data_count % bproplen == 0:
                loss.unchain_backward()  # Truncate the graph
            data_count += 1
        model.predictor.train = True
        return np.exp(float(sum_perp) / data_count)

    sum_perp = 0
    count = 0
    iteration = 0
    epoch_now = 0
    best_valid = 100000000
    state = None
    while train_iter.epoch < args.epoch:
        loss = 0
        iteration += 1
        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(args.bproplen):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()
            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = convert.concat_examples(batch, gpu_id)
            # Compute the loss at this time step and accumulate it
            state, loss_batch = optimizer.target(state, chainer.Variable(x), chainer.Variable(t))
            loss += loss_batch
            count += 1

        sum_perp += loss.data
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters

        if iteration % 100 == 0:
            logging.info('iteration: ' + str(iteration))
            logging.info('training perplexity: ' + str(np.exp(float(sum_perp) / count)))
            sum_perp = 0
            count = 0

        if train_iter.epoch > epoch_now:
            valid_perp = evaluate(model, valid_iter)
            logging.info('epoch: ' + str(train_iter.epoch))
            logging.info('validation perplexity: ' + str(valid_perp))

            # Save the model and the optimizer
            logging.info('save the model')
            serializers.save_npz(args.outdir + '/rnnlm.model.' + str(epoch_now), model)
            logging.info('save the optimizer')
            serializers.save_npz(args.outdir + '/rnnlm.state.' + str(epoch_now), optimizer)

            if valid_perp < best_valid:
                dest = args.outdir + '/rnnlm.model.best'
                if os.path.lexists(dest):
                    os.remove(dest)
                os.symlink('rnnlm.model.' + str(epoch_now), dest)
                best_valid = valid_perp

            epoch_now = train_iter.epoch
