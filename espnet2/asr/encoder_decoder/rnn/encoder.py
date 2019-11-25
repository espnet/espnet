import logging
from typing import Sequence, Tuple

import torch
from typeguard import typechecked
import numpy as np

from espnet.nets.chainer_backend.rnn.encoders import VGG2L, RNN
from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.initialization import \
    lecun_normal_init_parameters
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNNP
from espnet2.asr.encoder_decoder.abs_encoder import AbsEncoder


class Encoder(AbsEncoder):
    @typechecked
    def __init__(self,
                 idim: int,
                 etype: str = 'blstmp',
                 elayers: int = 4,
                 eunits: int = 300,
                 eprojs: int = 320,
                 dropout: float = 0.0,
                 subsample: Sequence[int] = None,
                 in_channel: int = 1):
        super().__init__()
        self.eprojs = eprojs

        typ = etype.lstrip("vgg").rstrip("p")
        if typ not in ['lstm', 'gru', 'blstm', 'bgru']:
            raise ValueError(
                "Error: need to specify an appropriate encoder architecture")

        if subsample is None:
            subsample = np.ones(elayers + 1, dtype=np.int)
        else:
            subsample = subsample[:elayers]
            # The first element is ignore and the second or later is used
            subsample = np.pad(np.array(subsample, dtype=np.int),
                               [1, elayers - len(subsample)], mode='constant',
                               constant_values=1)

        if etype.startswith("vgg"):
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [VGG2L(in_channel),
                     RNNP(get_vgg2l_odim(idim, in_channel=in_channel),
                          elayers, eunits, eprojs, subsample, dropout,
                          typ=typ)])
                logging.info('Use CNN-VGG + ' + typ.upper() + 'P for encoder')
            else:
                self.enc = torch.nn.ModuleList(
                    [VGG2L(in_channel),
                     RNN(get_vgg2l_odim(idim, in_channel=in_channel),
                         elayers, eunits, eprojs, dropout, typ=typ)])
                logging.info('Use CNN-VGG + ' + typ.upper() + ' for encoder')
        else:
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [RNNP(idim, elayers, eunits, eprojs, subsample, dropout,
                          typ=typ)])
                logging.info(
                    typ.upper() + ' with every-layer projection for encoder')
            else:
                self.enc = torch.nn.ModuleList(
                    [RNN(idim, elayers, eunits, eprojs, dropout, typ=typ)])
                logging.info(typ.upper() + ' without projection for encoder')

        self.init_like_chainer()

    def init_like_chainer(self):
        lecun_normal_init_parameters(self)

    def out_dim(self) -> int:
        return self.eprojs

    def forward(self, xs_pad: torch.Tensor, ilens: torch.Tensor,
                prev_states: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(
                xs_pad, ilens, prev_state=prev_state)
            current_states.append(states)

        xs_pad.masked_fill_(make_pad_mask(ilens, xs_pad, 1), 0.0),
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens, dtype=torch.long, device=xs_pad.device)
        # make mask to remove bias value in padded part
        return xs_pad, ilens, current_states
