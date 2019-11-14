from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.lm.lm_interface import LMInterface
from espnet2.utils.device_funcs import force_gatherable


class Model(torch.nn.Module):
    @typechecked
    def __init__(self, lm: LMInterface, ignore_id: int = -1):
        super().__init__()
        self.lm = lm
        self.ignore_id = ignore_id

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        # 0. Change pad_value
        input = input[:, :input_lengths.max()]
        mask = make_pad_mask(input_lengths).to(input.device)
        input.masked_fill_(mask, self.ignore_id)

        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # input: (Batch, Length) -> x, y: (Batch, Legnth + 1)
        x = torch.pad(input, [(1, 0), (0, 0)], 'constant', self.eos)
        t = torch.pad(input, [(0, 1), (0, 0)], 'constant', self.ignore_id)
        for l in input_lengths:
            t[l] = self.sos
        mask = torch.pad(mask, [(1, 0), (0, 0)], 'constant', True)

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y, _ = self.lm(x, None)

        # 3. Calc loss
        # loss: (BxL,)
        loss = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1),
                               reduction="none")

        # loss, mask: (BxL,) x (BxL,) -> (BxL,)
        loss = loss * mask.view(-1)
        # mask: (BxL,) -> ntokens: (1,)
        ntokens = mask.sum()
        # loss: (BxL,) -> (1,)
        loss = loss.sum() / ntokens
        stats = dict(loss=loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = \
            force_gatherable((loss, stats, ntokens), loss.device)
        return loss, stats, ntokens.detach()
