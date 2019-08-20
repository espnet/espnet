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
from espnet.nets.lm_interface import LMInterface

from espnet.utils.training.tensorboard_logger import TensorboardLogger
from tensorboardX import SummaryWriter

from espnet.utils.deterministic_utils import set_deterministic_chainer
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop


# TODO(karita): reimplement RNNLM with new interface
class DefaultRNNLM(LMInterface, link.Chain):
    """Default RNNLM wrapper to compute reduce framewise loss values.

    Args:
        n_vocab (int): The size of the vocabulary
        args (argparse.Namespace): configurations. see `add_arguments`
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--type', type=str, default="lstm", nargs='?', choices=['lstm', 'gru'],
                            help="Which type of RNN to use")
        parser.add_argument('--layer', '-l', type=int, default=2,
                            help='Number of hidden layers')
        parser.add_argument('--unit', '-u', type=int, default=650,
                            help='Number of hidden units')
        return parser


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
    :param str type: The RNN type
    """

    def __init__(self, n_vocab, n_layers, n_units, typ="lstm"):
        super(RNNLM, self).__init__()
        with self.init_scope():
            self.embed = DL.EmbedID(n_vocab, n_units)
            self.rnn = chainer.ChainList(
                *[L.StatelessLSTM(n_units, n_units) for _ in range(n_layers)]) if typ == "lstm" \
                else chainer.ChainList(*[L.StatelessGRU(n_units, n_units) for _ in range(n_layers)])
            self.lo = L.Linear(n_units, n_vocab)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.n_layers = n_layers
        self.n_units = n_units
        self.typ = typ

    def __call__(self, state, x):
        if state is None:
            if self.typ == "lstm":
                state = {'c': [None] * self.n_layers, 'h': [None] * self.n_layers}
            else:
                state = {'h': [None] * self.n_layers}

        h = [None] * self.n_layers
        emb = self.embed(x)
        if self.typ == "lstm":
            c = [None] * self.n_layers
            c[0], h[0] = self.rnn[0](state['c'][0], state['h'][0], F.dropout(emb))
            for n in six.moves.range(1, self.n_layers):
                c[n], h[n] = self.rnn[n](state['c'][n], state['h'][n], F.dropout(h[n - 1]))
            state = {'c': c, 'h': h}
        else:
            if state['h'][0] is None:
                xp = self.xp
                with chainer.backends.cuda.get_device_from_id(self._device_id):
                    state['h'][0] = chainer.Variable(
                        xp.zeros((emb.shape[0], self.n_units), dtype=emb.dtype))
            h[0] = self.rnn[0](state['h'][0], F.dropout(emb))
            for n in six.moves.range(1, self.n_layers):
                if state['h'][n] is None:
                    xp = self.xp
                    with chainer.backends.cuda.get_device_from_id(self._device_id):
                        state['h'][n] = chainer.Variable(
                            xp.zeros((h[n - 1].shape[0], self.n_units), dtype=h[n - 1].dtype))
                h[n] = self.rnn[n](state['h'][n], F.dropout(h[n - 1]))
            state = {'h': h}
        y = self.lo(F.dropout(h[-1]))
        return state, y