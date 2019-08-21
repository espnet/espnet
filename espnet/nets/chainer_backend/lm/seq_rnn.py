# Copyright 2019 Waseda University (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code is ported from the following implementation written in chainer.
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


class SequentialRNNLM(LMInterface, link.Chain):
    """Sequential RNNLM.

    See also:
        https://github.com/pytorch/examples/blob/4581968193699de14b56527296262dd76ab43557/word_language_model/model.py

    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--type', type=str, default="lstm", nargs='?', choices=['lstm', 'gru'],
                            help="Which type of RNN to use")
        parser.add_argument('--layer', '-l', type=int, default=2,
                            help='Number of hidden layers')
        parser.add_argument('--unit', '-u', type=int, default=650,
                            help='Number of hidden units')
        parser.add_argument('--dropout-rate', type=float, default=0.5,
                            help='dropout probability')
        return parser
    
    def __init__(self, n_vocab, args):
        chainer.Chain.__init__(self)
        self._setup(
            rnn_type=args.type.upper(),
            ntoken=n_vocab,
            ninp=args.unit,
            nhid=args.unit,
            nlayers=args.layer,
            dropout=args.dropout_rate)

    def _setup(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        with self.init_scope():
            self.encoder = DL.EmbedID(ntoken, ninp)
            if rnn_type in ['LSTM', 'GRU']:
                rnn = L.NStepLSTM if rnn_type == "lstm" else L.NStepGRU
                
            else:
                try:
                    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                except KeyError:
                    raise ValueError("""An invalid option for `--model` was supplied,
                                    options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
                rnn = L.NStepRNNTanh if nonlinearity == "tanh" else L.NStepRNNReLU
            self.rnn = rnn(nlayers, ninp, nhid, dropout)
            self.decoder = L.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.W = self.encoder.W

        self._init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def _init_weights(self):
        # NOTE: original init in pytorch/examples
        # initrange = 0.1
        # self.encoder.W.data.uniform_(-initrange, initrange)
        # self.decoder.b.data.zero_()
        # self.decoder.W.data.uniform_(-initrange, initrange)
        # NOTE: our default.py:RNNLM init
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def forward(self, x, t):
        y = self._before_loss(x, None)[0]
        mask = (x != 0).astype(y.dtype)
        loss = F.softmax_cross_entropy(y.reshape(-1, y.shape[-1]), t.reshape(-1), reduce="no")
        logp = loss * mask.reshape(-1)
        logp = F.sum(logp)
        count = mask.sum()
        return logp / count, logp, count

    def _before_loss(self, input, hidden):
        emb = self.encoder(input)
        hidden, output = self.rnn(hidden, F.separate(emb, axis=0))
        decoded = self.decoder(F.stack(output, axis=0), n_batch_axes=2)
        return decoded, hidden

    def score(self, y, state, x):
        """Score new token.

        Args:
            y (xp.ndarray): 1D xp.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (xp.ndarray): 2D encoder feature that generates ys.

        Returns:
            tuple[xp.ndarray, Any]: Tuple of
                xp.float32 scores for next token (n_vocab)
                and next state for ys

        """
        y, new_state = self._before_loss(y[-1].reshape(1, 1), state)
        logp = F.log_softmax(y, axis=1).reshape(-1)
        return logp, new_state
