import logging

import torch
import numpy as np

from espnet.nets.chainer_backend.rnn.encoders import VGG2L, RNN
from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.rnn.encoders import RNNP


class Encoder(torch.nn.Module):
    def __init__(self,
                 idim: int = 80,
                 etype: str = 'blstmp',
                 elayers: int = 4,
                 eunits: int = 300,
                 eprojs: int = 256,
                 dropout: float = 0.0,
                 in_channel: int = 1):
        super(Encoder, self).__init__()

        typ = etype.lstrip("vgg").rstrip("p")
        if typ not in ['lstm', 'gru', 'blstm', 'bgru']:
            raise ValueError(
                "Error: need to specify an appropriate encoder architecture")

        subsample = np.ones(elayers + 1, dtype=np.int)
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

    def forward(self, xs_pad, ilens, prev_states=None):
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(
                xs_pad, ilens, prev_state=prev_state)
            current_states.append(states)

        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))
        return xs_pad.masked_fill(mask, 0.0), ilens, current_states
