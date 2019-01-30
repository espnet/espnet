import logging
import sys

import numpy as np
import six
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from espnet.nets.e2e_asr_common import expand_elayers
from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device


class BLSTMP(torch.nn.Module):
    """Bidirectional LSTM with projection layer module

    :param int idim: dimension of inputs
    :param list[tuple[int,float,int,float]] elayers: layers configuration
    :param np.ndarray subsample: list of subsampling numbers
    """

    def __init__(self, idim, elayers, subsample):
        super(BLSTMP, self).__init__()
        for layer in six.moves.range(len(elayers)):
            units = elayers[layer][0]
            projs = elayers[layer][2]
            if layer == 0:
                inputdim = idim
            else:
                inputdim = elayers[layer - 1][2]
            # Must do dropout explicitely https://github.com/espnet/espnet/issues/259
            setattr(self, "bilstm%d" % layer,
                    torch.nn.LSTM(inputdim, units, num_layers=1, bidirectional=True, batch_first=True))
            # bottleneck layer to merge
            setattr(self, "bt%d" % layer, torch.nn.Linear(2 * units, projs))

        self.elayers = elayers
        self.subsample = subsample

    def forward(self, xs_pad, ilens):
        """BLSTMP forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, hdim)
        :rtype: torch.Tensor
        """
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        for layer in six.moves.range(len(self.elayers)):
            xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
            bilstm = getattr(self, 'bilstm' + str(layer))
            bilstm.flatten_parameters()
            ys, _ = bilstm(xs_pack)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
            dropout = self.elayers[layer][1]
            if dropout > 0:
                ys_pad = F.dropout(ys_pad, dropout)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ys_pad = ys_pad[:, ::sub]
                ilens = [int(i + 1) // sub for i in ilens]
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'bt' + str(layer)
                                )(ys_pad.contiguous().view(-1, ys_pad.size(2)))
            proj_dropout = self.elayers[layer][3]
            if proj_dropout > 0:
                projected = F.dropout(projected, proj_dropout)
            xs_pad = torch.tanh(projected.view(ys_pad.size(0), ys_pad.size(1), -1))
            proj_dropout = self.elayers[layer][3]
            if proj_dropout > 0:
                xs_pad = F.dropout(xs_pad, proj_dropout)

        return xs_pad, ilens  # x: utt list of frame x dim


class BLSTM(torch.nn.Module):
    """Bidirectional LSTM module

    :param int idim: dimension of inputs
    :param list[tuple[int,float,int,float]] elayers: layers configuration
    """

    def __init__(self, idim, elayers):
        super(BLSTM, self).__init__()
        num_layers = len(elayers)
        cdim = elayers[0][0]
        dropout = elayers[0][1]
        hdim = elayers[0][2]
        self.nblstm = torch.nn.LSTM(idim, cdim, num_layers, batch_first=True,
                                    dropout=dropout, bidirectional=True)
        self.l_last = torch.nn.Linear(cdim * 2, hdim)
        self.proj_dropout = elayers[0][3]

    def forward(self, xs_pad, ilens):
        """BLSTM forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        self.nblstm.flatten_parameters()
        ys, _ = self.nblstm(xs_pack)
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
        if self.nblstm.dropout > 0:
            ys_pad = F.dropout(ys_pad, self.nblstm.dropout)
        # (sum _utt frame_utt) x dim
        projected = torch.tanh(self.l_last(
            ys_pad.contiguous().view(-1, ys_pad.size(2))))
        if self.proj_dropout > 0:
            projected = F.dropout(projected, self.proj_dropout)
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)

        return xs_pad, ilens  # x: utt list of frame x dim


class VGG2L(torch.nn.Module):
    """VGG-like module

    :param int in_channel: number of input channels
    """

    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.in_channel = in_channel

    def forward(self, xs_pad, ilens):
        """VGG2L forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of padded hidden state sequences (B, Tmax // 4, 128)
        :rtype: torch.Tensor
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        # xs_pad = F.pad_sequence(xs_pad)

        # x: utt x 1 (input channel num) x frame x dim
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), self.in_channel,
                             xs_pad.size(2) // self.in_channel).transpose(1, 2)

        # NOTE: max_pool1d ?
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)

        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        return xs_pad, ilens


class Encoder(torch.nn.Module):
    """Encoder module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param str elayers: Encoder layers configuration
    :param np.ndarray subsample: list of subsampling numbers
    :param int in_channel: number of input channels
    """

    def __init__(self, etype, idim, elayers, subsample, in_channel=1):
        super(Encoder, self).__init__()
        expanded_elayers, etype = expand_elayers(elayers, etype, warn=True)
        if etype == 'blstm':
            self.enc = torch.nn.ModuleList([BLSTM(idim, expanded_elayers)])
            logging.info('BLSTM without projection for encoder')
        elif etype == 'blstmp':
            self.enc = torch.nn.ModuleList([BLSTMP(idim, expanded_elayers, subsample)])
            logging.info('BLSTM with every-layer projection for encoder')
        elif etype == 'vggblstmp':
            self.enc = torch.nn.ModuleList([VGG2L(in_channel),
                                            BLSTMP(get_vgg2l_odim(idim, in_channel=in_channel), expanded_elayers,
                                                   subsample)])
            logging.info('Use CNN-VGG + BLSTMP for encoder')
        elif etype == 'vggblstm':
            self.enc = torch.nn.ModuleList([VGG2L(in_channel),
                                            BLSTM(get_vgg2l_odim(idim, in_channel=in_channel), expanded_elayers)])
            logging.info('Use CNN-VGG + BLSTM for encoder')
        else:
            logging.error(
                "Error: need to specify an appropriate encoder architecture")
            sys.exit()

    def forward(self, xs_pad, ilens):
        """Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        for module in self.enc:
            xs_pad, ilens = module(xs_pad, ilens)

        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad.masked_fill(mask, 0.0), ilens


def encoder_for(args, idim, subsample):
    return Encoder(args.etype, idim, args.elayers, subsample)
