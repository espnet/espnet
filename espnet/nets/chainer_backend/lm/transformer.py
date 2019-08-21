"""Transformer language model."""

# Copyright 2019 Waseda University (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import numpy as np

import chainer.functions as F
import chainer.links as L

from chainer import link

from espnet.nets.chainer_backend.transformer.embedding import PositionalEncoding
from espnet.nets.chainer_backend.transformer.encoder import Encoder
from espnet.nets.lm_interface import LMInterface


# TODO(karita): reimplement RNNLM with new interface
class TransformerLM(LMInterface, link.Chain):
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
        self.model_type = 'Transformer'
        self.src_mask = None
        with self.init_scope():
            self.encoder = Encoder(
                n_vocab, args.att_unit, args.head, args.unit, args.layer,
                args.dropout_rate, args.dropout_rate, args.dropout_rate,
                input_layer="embed")
            # reset posenc
            self.encoder.input_layer[1] = PositionalEncoding(args.att_unit, args.dropout_rate, args.posenc_len)
            self.decoder = L.Linear(args.att_unit, n_vocab)

    def forward(self, x, t):
        """Compute LM loss value from buffer sequences.

        Args:
            x (ndarray): Input ids. (batch, len)
            t (ndarray): Target ids. (batch, len)

        Returns:
            tuple[chainer.Variable, chainer.Variable, int]: Tuple of
                loss to backward (scalar),
                negative log-likelihood of t: -log p(t) (scalar) and
                the number of elements in x (scalar)

        Notes:
            The last two return values are used in perplexity: p(t)^{-n} = exp(-log p(t) / n)

        """
        xp = self.xp
        xm = (x != 0)
        to_array = xp.asarray if xp is np else xp.asnumpy
        ilens = np.sum(to_array(xm), axis=1)
        h, _, _ = self.encoder(x, ilens)
        y = self.decoder(h, n_batch_axes=2)
        loss = F.softmax_cross_entropy(y.reshape(-1, y.shape[-1]), t.reshape(-1), reduce="no")
        mask = xm.astype(y.dtype)
        logp = loss * mask.reshape(-1)
        logp = F.sum(logp)
        count = mask.sum()
        return logp / count, logp, count

    def score(self, y, state, x):
        """Score new token.

        Args:
            y (ndarray): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (ndarray): encoder feature that generates ys.

        Returns:
            tuple[ndarray, Any]: Tuple of
                torch.float32 scores for next token (n_vocab)
                and next state for ys

        """
        y = F.expand_dims(y, axis=0)
        h, _, _ = self.encoder(y, y.shape)
        h = self.decoder(h)[:, -1]
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, None
