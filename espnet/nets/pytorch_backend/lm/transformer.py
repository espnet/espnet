from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.lm_interface import LMInterface
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask


class TransformerLM(nn.Module, LMInterface):

    @staticmethod
    def add_arguments(parser):
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
        return parser

    def __init__(self, n_vocab, args):
        nn.Module.__init__(self)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.encoder = Encoder(
            n_vocab, args.att_unit, args.head, args.unit, args.layer,
            args.dropout_rate, args.dropout_rate, args.dropout_rate,
            input_layer="embed")
        self.decoder = nn.Linear(args.att_unit, n_vocab)

    def target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        xm = (x != 0)
        h, _ = self.encoder(x, self.target_mask(x))
        y = self.decoder(h)
        loss = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1), reduction="none")
        mask = xm.to(dtype=loss.dtype)
        logp = loss * mask.view(-1)
        logp = logp.sum()
        count = mask.sum()
        return logp / count, logp, count

    def score(self, y: torch.Tensor, state: Any, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        y = y.unsqueeze(0)
        h, _ = self.encoder(y, self.target_mask(y))
        h = self.decoder(h)[:, -1]
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, None
