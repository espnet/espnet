"""Transformer language model."""

from typing import Any
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.lm_interface import LMInterface
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask


class TransformerLM(nn.Module, LMInterface):
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
        nn.Module.__init__(self)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.encoder = Encoder(
            n_vocab, args.att_unit, args.head, args.unit, args.layer,
            args.dropout_rate, args.dropout_rate, args.dropout_rate,
            input_layer="embed")
        # reset posenc
        self.encoder.embed[1] = PositionalEncoding(args.att_unit, args.dropout_rate, args.posenc_len)
        self.decoder = nn.Linear(args.att_unit, n_vocab)

    def _target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute LM loss value from buffer sequences.

        Args:
            x (torch.Tensor): Input ids. (batch, len)
            t (torch.Tensor): Target ids. (batch, len)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of
                loss to backward (scalar),
                negative log-likelihood of t: -log p(t) (scalar) and
                the number of elements in x (scalar)

        Notes:
            The last two return values are used in perplexity: p(t)^{-n} = exp(-log p(t) / n)

        """
        xm = (x != 0)
        h, _ = self.encoder(x, self._target_mask(x))
        y = self.decoder(h)
        loss = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1), reduction="none")
        mask = xm.to(dtype=loss.dtype)
        logp = loss * mask.view(-1)
        logp = logp.sum()
        count = mask.sum()
        return logp / count, logp, count

    def score(self, y: torch.Tensor, state: Any, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (n_vocab)
                and next state for ys

        """
        y = y.unsqueeze(0)
        h, _ = self.encoder(y, self._target_mask(y))
        h = self.decoder(h)[:, -1]
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, None
