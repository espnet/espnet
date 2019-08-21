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

from espnet.utils.training.tensorboard_logger import TensorboardLogger
from tensorboardX import SummaryWriter

from espnet.utils.deterministic_utils import set_deterministic_chainer
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

from espnet.nets.lm_interface import LMInterface
from espnet.nets.chainer_backend.transformer.embedding import PositionalEncoding
from espnet.nets.chainer_backend.transformer.encoder import Encoder
from espnet.nets.chainer_backend.transformer.mask import make_history_mask


# TODO(karita): reimplement RNNLM with new interface
class DefaultRNNLM(LMInterface, link.Chain):
    """Transformer language model."""

    @staticmethod
    def add_arguments(parser):
        """Add arguments to command line argument parser."""
        parser.add_argument('--layer', type=int, default=4,
                            help='Number of hidden layers')
        parser.add_argument('--unit', type=int, default=1024,
                            help='Number of hidden units in feedforward layer')
        parser.add_argument('--att-unit', type=int, default=256,
                            help='Number of hidden units in attention layer')
        parser.add_argument('--head', type=int, default=2,
                            help='Number of multi head attention')
        parser.add_argument('--dropout-rate', type=float, default=0.5,
                            help='dropout probability')
        parser.add_argument('--posenc-len', type=int, default=10000,
                            help='Predefined length of positional encoding cache')
        return parser

    def __init__(self, n_vocab, args):
        """Initialize class.

        Args:
            n_vocab (int): The size of the vocabulary
            args (argparse.Namespace): configurations. see py:method:`add_arguments`

        """
        link.Chain.__init__(self)
        with self.init_scope():

    
    def forward(self, x, t) -> Tuple[chainer.Variable, chainer.Variable, int]:
        xm = (x != 0)
