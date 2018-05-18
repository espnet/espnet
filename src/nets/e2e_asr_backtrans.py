#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import six

import numpy as np
import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


def pad_list(batch, pad_value=0.0):
    """FUNCTION TO PAD VALUE

    :param list batch: list of the sequences, where the shape of i-th sequence (T_i, C)
    :param float pad_value: value to pad

    :return: padded batch with the shape (B, T_max, C)
    """
    bs = len(batch)
    if isinstance(batch[0], Variable):
        # for pytorch variable
        maxlen = max(x.size(0) for x in batch)
        batch_pad = Variable(batch[0].data.new(
            bs, maxlen, *batch[0].size()[1:]).zero_() + pad_value)
        for i in range(bs):
            batch_pad[i, :batch[i].size(0)] = batch[i]
    else:
        # for numpy ndarray
        maxlen = max([b.shape[0] for b in batch])
        if len(batch[0].shape) >= 2:
            batch_pad = np.zeros((bs, maxlen, *batch[0].shape[1:]))
        else:
            batch_pad = np.zeros((bs, maxlen))
        for idx, batch in enumerate(batch):
            batch_pad[idx, :batch.shape[0]] = batch

    return batch_pad


class BackTranslater(torch.nn.Module):
    def __init__(self, idim, odim, elayers=256, eunits=1, dlayers=2, dunits=1024,
                 atype="NoAtt", threshold=0.5, minlenratio=0.0, maxlenratio=0.0):
        super(BackTranslater, self).__init__()
        self.idim = idim
        self.odim = odim
        self.elayers = elayers
        self.eunits = eunits
        self.dlayers = dlayers
        self.dunits = dunits
        self.atype = atype
        self.threshold = threshold
        self.minlenratio = minlenratio
        self.maxlenratio = maxlenratio
        self.enc = CharEncoder(
            self.idim, 512, elayers, eunits)
        att = NoAtt()
        self.dec = FeatureDecoder(
            eunits * 2, odim, att, dlayers, dunits)

    def forward(self, xs, xlens, ys, ylens):
        hs, hlens = self.enc(xs, xlens)
        outs, probs, olens = self.dec(hs, hlens, ys, ylens)
        mse_loss = 0.0
        bc_loss = 0.0
        for y, out, prob, olen in zip(ys, outs, probs, olens):
            mse_loss += torch.nn.functional.mse_loss(
                out[:olen], y[:olen], size_average=False)
            target = torch.zeros(olen).float()
            target[-1] = 1.0
            bc_loss += torch.nn.functional.binary_cross_entropy(
                F.sigmoid(prob[:olen]), target, size_average=False)
        mse_loss /= sum(olens)
        bc_loss /= sum(olens)

        return mse_loss, bc_loss

    def predict(self, x):
        # setup
        xs = x.unsqueeze(0)
        xlens = [x.size(0)]
        maxlen = int(x.size(0) * self.maxlenratio)
        minlen = int(x.size(0) * self.minlenratio)

        hs, hlens = self.enc(xs, xlens)

        # initialize hidden states of decoder
        c_list = [self.dec.zero_state(hs)]
        z_list = [self.dec.zero_state(hs)]
        for l in six.moves.range(1, self.dlayers):
            c_list += [self.dec.zero_state(hs)]
            z_list += [self.dec.zero_state(hs)]
        prev_out = self.dec.feat_out(z_list[-1])

        # initialize attention
        att_w = None
        self.dec.att.reset()

        # loop for an output sequence
        idx = 0
        outs = []
        while True:
            # updated index
            idx += 1

            # decoder calculation
            att_c, att_w = self.dec.att(hs, hlens, z_list[0], att_w)
            prenet_out = self.dec.prenet(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.dec.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.dec.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            outs += [self.dec.feat_out(z_list[-1])]
            prob = F.sigmoid(self.dec.prob_out(z_list[-1]))[0]

            # check whether to finish generation
            if (prob >= self.thres and idx >= minlen) or idx == maxlen:
                outs = torch.stack(outs, dim=2)  # (1, odim, Lmax)
                outs += self.dec.postnet(outs)  # (1, odim, Lmax)
                outs = outs.transpose(2, 1).unsqueeze(0)  # (Lmax, odim)
                return outs


class CharEncoder(torch.nn.Module):
    """CHARACTER EMBEDDING ENCODER

    This is the encoder which converts the sequence of characters into
    the sequence of hidden states. The newtwork structure is based on
    that of tacotron2 in the field of speech synthesis.

    :param int idim: dimension of the inputs
    :param int edim: dimension of character embedding
    :param int elayers: the number of blstm layers
    :param int eunits: the number of blstm units
    :param float dropout: dropout rate in lstm layer
    """

    def __init__(self, idim, edim=512, elayers=1, eunits=256, dropout=0.0):
        super(CharEncoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.edim = edim
        self.elayers = elayers
        self.eunits = eunits
        self.dropout = dropout
        # define network layer modules
        self.embed = torch.nn.Embedding(idim, edim)
        self.convs = torch.nn.Sequential(
            torch.nn.Conv1d(edim, edim, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(edim),
            torch.nn.ReLU(),
            torch.nn.Conv1d(edim, edim, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(edim),
            torch.nn.ReLU(),
            torch.nn.Conv1d(edim, edim, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(edim),
            torch.nn.ReLU(),
        )
        self.blstm = torch.nn.LSTM(
            edim, eunits, elayers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, xs, ilens):
        """CHARACTER ENCODER FORWARD CALCULATION

        :param tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each batch (B)

        :return: batch of sequences of padded encoder states (B, Tmax, eunits*2)
        :rtype: tensor
        :return: list of lengths of each batch (B)
        :rtype: list
        """
        out = self.embed(xs)  # (B, Tmax, C)
        out = self.convs(out.transpose(1, 2))  # (B, C, Tmax)
        out = pack_padded_sequence(out.transpose(1, 2), ilens, batch_first=True)
        out, _ = self.blstm(out)  # (B, Tmax, C)
        out, olens = pad_packed_sequence(out, batch_first=True)

        return out, olens.tolist()


class NoAtt(torch.nn.Module):
    '''No attention'''

    def __init__(self):
        super(NoAtt, self).__init__()
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''NoAtt forward

        :param Variable enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param Variable dec_z: dummy (does not use)
        :param Variable att_prev: dummy (does not use)
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: Variable
        :return: previous attentioin weights
        :rtype: Variable
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [Variable(enc_hs_pad.data.new(
                l).zero_() + (1.0 / l)) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev = pad_list(att_prev, 0)
            self.c = torch.sum(self.enc_h * att_prev.view(batch, self.h_length, 1), dim=1)

        return self.c, att_prev


class FeatureDecoder(torch.nn.Module):
    """DECODER TO PREDICT THE SEQUENCE OF FEATURES

    This the decoder which generate the sequence of features from
    the sequence of the hidden states. The network structure is
    based on that of the tacotron2 in the field of speech synthesis.

    :param int idim: # dimensions of the inputs
    :param int odim: # dimensions of the outputs
    :param instance att: instance of attetion class
    :param int dlayers: # layers of decoder lstm
    :param int dunits: # units of decoder lstm
    """

    def __init__(self, idim, odim, att, dlayers=2, dunits=1024):
        super(FeatureDecoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.odim = odim
        self.att = att
        self.dlayers = dlayers
        self.dunits = dunits
        # define the network layer modules
        self.lstm = torch.nn.ModuleList()
        self.lstm += [torch.nn.LSTMCell(idim + 256, dunits)]
        for dl in six.moves.range(1, dlayers):
            self.lstm += [torch.nn.LSTMCell(dunits, dunits)]
        self.prenet = torch.nn.Sequential(
            torch.nn.Linear(odim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.postnet = torch.nn.Sequential(
            torch.nn.Conv1d(odim, 512, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.Tanh(),
            torch.nn.Conv1d(512, 512, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.Tanh(),
            torch.nn.Conv1d(512, 512, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.Tanh(),
            torch.nn.Conv1d(512, 512, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.Tanh(),
            torch.nn.Conv1d(512, odim, 5, stride=1, padding=2, bias=False),
        )
        # use 1x1 conv instead of linear
        self.feat_out = torch.nn.Linear(dunits, odim)
        self.prob_out = torch.nn.Linear(dunits, 1)

    def zero_state(self, hs):
        return Variable(hs.data.new(hs.size(0), self.dunits).zero_())

    def forward(self, hs, hlens, ys, ylens):
        """DECODER FORWARD CALCULATION

        :param tensor hs: batch of the sequences of padded hidden states (B, Tmax, idim)
        :param list hlens: list of lengths of each input batch (B)
        :param tensor ys: batch of the sequences of padded target features (B, Lmax, odim)
        :param list ylens: list of lengths of each output batch (B)
        """
        # initialize hidden states of decoder
        c_list = [self.zero_state(hs)]
        z_list = [self.zero_state(hs)]
        for l in six.moves.range(1, self.dlayers):
            c_list += [self.zero_state(hs)]
            z_list += [self.zero_state(hs)]
        prev_out = self.feat_out(z_list[-1])

        # initialize attention
        att_w = None
        self.att.reset()

        # loop for an output sequence
        outs = []
        probs = []
        for i in six.moves.range(max(ylens)):
            att_c, att_w = self.att(hs, hlens, z_list[0], att_w)
            prenet_out = self.prenet(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            outs += [self.feat_out(z_list[-1])]
            probs += [self.prob_out(z_list[-1])]
            prev_out = outs[-1]

        outs = torch.stack(outs, dim=2)  # (B, odim, Lmax)
        probs = torch.cat(probs, dim=1)  # (B, Lmax)
        outs += self.postnet(outs)  # (B, odim, Lmax)
        outs = outs.transpose(2, 1)  # (B, Lmax, odim)

        return outs, probs, ylens
