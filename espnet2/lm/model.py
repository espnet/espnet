from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.lm.abs_lm import AbsLM
from espnet2.train.abs_espnet_model import AbsESPNetModel
from espnet2.utils.device_funcs import force_gatherable


class LanguageModel(AbsESPNetModel):
    @typechecked
    def __init__(self, lm: AbsLM, sos_and_eos: int, ignore_id: int = 0):
        super().__init__()
        self.lm = lm
        self.sos = sos_and_eos
        self.eos = sos_and_eos

        # ignore_id may be assumed as 0, shared with CTC-blank symbol for ASR.
        # in the other part.
        self.ignore_id = ignore_id

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        assert input.size(-1) >= input_lengths.max(), (input.size(),
                                                       input_lengths.max())
        # 0. Change pad_value
        input = input[:, :input_lengths.max()]
        mask = make_pad_mask(input_lengths).to(input.device)
        input.masked_fill_(mask, self.ignore_id)

        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # input: (Batch, Length) -> x, y: (Batch, Legnth + 1)
        x = F.pad(input, [1, 0], 'constant', self.eos)
        t = F.pad(input, [0, 1], 'constant', self.ignore_id)
        mask = F.pad(mask, [0, 1], 'constant', True)
        for i, l in enumerate(input_lengths):
            t[i, l] = self.sos

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y, _ = self.lm(x, None)

        # 3. Calc loss
        # loss: (BxL,)
        loss = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1),
                               reduction="none")

        # loss, mask: (BxL,) x (BxL,) -> loss: (BxL,)
        loss.masked_fill_(mask.view(-1), 0.)
        # mask: (BxL,) -> ntokens: (1,)
        ntokens = (~mask).sum()
        # loss: (BxL,) -> (1,)
        loss = loss.sum() / ntokens
        stats = dict(loss=loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = \
            force_gatherable((loss, stats, ntokens), loss.device)
        return loss, stats, weight
