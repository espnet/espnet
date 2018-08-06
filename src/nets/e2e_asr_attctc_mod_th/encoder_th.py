#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging
import math
import sys

import numpy as np
import six
import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

# from e2e_asr_attctc_mod_th.model import pad_list
# from e2e_asr_attctc_mod_th.model import torch_is_old
from e2e_asr_attctc_th import pad_list
from e2e_asr_attctc_th import torch_is_old


# get output dim for latter BLSTM
def _get_vgg2l_odim(idim, in_channel=3, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels


# get output dim for latter BLSTM
def _get_max_pooled_size(idim, out_channel=128, n_layers=2, ksize=2, stride=2):
    for _ in six.moves.range(n_layers):
        idim = math.floor((idim - (ksize - 1) - 1) / stride)
    return idim  # numer of channels


class Encoder(torch.nn.Module):
    '''ENCODER NETWORK CLASS

    This is the example of docstring.

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int epojs: number of projection units of encoder network
    :param str subsample: subsampling number e.g. 1_2_2_2_1
    :param float dropout: dropout rate
    :return:

    '''

    def __init__(self, etype, idim, elayers, eunits, eprojs, subsample, dropout, in_channel=1):
        super(Encoder, self).__init__()

        if etype == 'blstm':
            self.enc1 = BLSTM(idim, elayers, eunits, eprojs, dropout)
            logging.info('BLSTM without projection for encoder')
        elif etype == 'blstmp':
            self.enc1 = BLSTMP(idim, elayers, eunits,
                               eprojs, subsample, dropout)
            logging.info('BLSTM with every-layer projection for encoder')
        elif etype == 'vggblstmp':
            self.enc1 = VGG2L(in_channel)
            self.enc2 = BLSTMP(_get_vgg2l_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            logging.info('Use CNN-VGG + BLSTMP for encoder')
        elif etype == 'vggblstm':
            self.enc1 = VGG2L(in_channel)
            self.enc2 = BLSTM(_get_vgg2l_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG + BLSTM for encoder')
        else:
            logging.error(
                "Error: need to specify an appropriate encoder archtecture")
            sys.exit()

        self.etype = etype

    def forward(self, xs, ilens):
        '''Encoder forward

        :param xs:
        :param ilens:
        :return:
        '''
        if self.etype == 'blstm':
            xs, ilens = self.enc1(xs, ilens)
        elif self.etype == 'blstmp':
            xs, ilens = self.enc1(xs, ilens)
        elif self.etype == 'vggblstmp':
            xs, ilens = self.enc1(xs, ilens)
            xs, ilens = self.enc2(xs, ilens)
        elif self.etype == 'vggblstm':
            xs, ilens = self.enc1(xs, ilens)
            xs, ilens = self.enc2(xs, ilens)
        else:
            logging.error(
                "Error: need to specify an appropriate encoder archtecture")
            sys.exit()

        return xs, ilens


class BLSTMP(torch.nn.Module):
    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout):
        super(BLSTMP, self).__init__()
        for i in six.moves.range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim
            setattr(self, "bilstm%d" % i, torch.nn.LSTM(inputdim, cdim, dropout=dropout,
                                                        num_layers=1, bidirectional=True, batch_first=True))
            # Dropout for hidden-hidden or hidden-output connection
            setattr(self, 'd_bilstm' + str(i), torch.nn.Dropout(p=dropout))
            # bottleneck layer to merge
            setattr(self, "bt%d" % i, torch.nn.Linear(2 * cdim, hdim))
            setattr(self, 'd_p' + str(i), torch.nn.Dropout(p=dropout))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample

    def forward(self, xpad, ilens):
        '''BLSTMP forward

        :param xs:
        :param ilens:
        :return:
        '''
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        for layer in six.moves.range(self.elayers):
            xpack = pack_padded_sequence(xpad, ilens, batch_first=True)
            bilstm = getattr(self, 'bilstm' + str(layer))
            if torch_is_old:
                # pytorch 0.4.x does not support flatten_parameters() for multiple GPUs
                bilstm.flatten_parameters()
            ys, (hy, cy) = bilstm(xpack)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ypad, ilens = pad_packed_sequence(ys, batch_first=True)
            ypad = getattr(self, 'd_bilstm' + str(layer))(ypad)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ypad = ypad[:, ::sub]
                ilens = [int(i + 1) // sub for i in ilens]
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'bt' + str(layer)
                                )(ypad.contiguous().view(-1, ypad.size(2)))
            projected = getattr(self, 'd_p' + str(layer))(projected)
            xpad = torch.tanh(projected.view(ypad.size(0), ypad.size(1), -1))
            del hy, cy

        return xpad, ilens  # x: utt list of frame x dim


class BLSTM(torch.nn.Module):
    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(BLSTM, self).__init__()
        self.nblstm = torch.nn.LSTM(idim, cdim, elayers, batch_first=True,
                                    dropout=dropout, bidirectional=True)
        self.l_last = torch.nn.Linear(cdim * 2, hdim)

    def forward(self, xpad, ilens):
        '''BLSTM forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        xpack = pack_padded_sequence(xpad, ilens, batch_first=True)
        ys, (hy, cy) = self.nblstm(xpack)
        del hy, cy
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ypad, ilens = pad_packed_sequence(ys, batch_first=True)
        # (sum _utt frame_utt) x dim
        projected = torch.tanh(self.l_last(
            ypad.contiguous().view(-1, ypad.size(2))))
        xpad = projected.view(ypad.size(0), ypad.size(1), -1)
        return xpad, ilens  # x: utt list of frame x dim


class VGG2L(torch.nn.Module):
    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.in_channel = in_channel

    def forward(self, xs, ilens):
        '''VGG2L forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        # xs = F.pad_sequence(xs)

        # x: utt x 1 (input channel num) x frame x dim
        xs = xs.view(xs.size(0), xs.size(1), self.in_channel,
                     xs.size(2) // self.in_channel).transpose(1, 2)

        # NOTE: max_pool1d ?
        xs = F.relu(self.conv1_1(xs))
        xs = F.relu(self.conv1_2(xs))
        xs = F.max_pool2d(xs, 2, stride=2, ceil_mode=True)

        xs = F.relu(self.conv2_1(xs))
        xs = F.relu(self.conv2_2(xs))
        xs = F.max_pool2d(xs, 2, stride=2, ceil_mode=True)
        # change ilens accordingly
        # ilens = [_get_max_pooled_size(i) for i in ilens]
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = xs.transpose(1, 2)
        xs = xs.contiguous().view(
            xs.size(0), xs.size(1), xs.size(2) * xs.size(3))
        xs = [xs[i, :ilens[i]] for i in six.moves.range(len(ilens))]
        xs = pad_list(xs, 0.0)
        return xs, ilens
