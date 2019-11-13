from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.lm.lm_interface import LMInterface


class E2E(torch.nn.Module):
    @typechecked
    def __init__(self, lm: LMInterface, ignore_id: int = -1):
        super().__init__()
        self.lm = lm
        self.ignore_id = ignore_id

    def forward(self, input: torch.Tesnor, input_lengths: torch.Tensor):
        # 0. Change pad_value
        mask = make_pad_mask(input_lengths, xs=input)
        input.masked_fill_(mask, self.ignore_id)

        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # input: (Batch, Length) -> x, y: (Batch, Legnth + 1)
        x = torch.pad(input, [(1, 0), (0, 0)], 'constant', self.eos)
        t = torch.pad(input, [(0, 0), (0, 1)], 'constant', self.ignore_id)
        for l in input_lengths:
            t[l] = self.sos
        mask = torch.pad(mask, [(1, 0), (0, 0)], 'constant', True)

        # 3. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y, _ = self.lm(x, None)
        # loss: (BxL,)
        loss = F.cross_entropy(
            y.view(-1, y.shape[-1]), t.view(-1), reduction="none")

        # loss, mask: (BxL,) x (BxL,) -> (BxL,)
        loss = loss * mask.view(-1)
        # mask: (BxL,) -> count: (1,)
        count = mask.sum()
        # loss: (Batch,) -> (1,)
        loss = loss.sum() / count
        stats = dict(
            loss=loss.detach(),
            count=count.detach())

        return loss, stats
