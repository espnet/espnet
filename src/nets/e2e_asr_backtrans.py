#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import six

import chainer
import torch
import torch.nn.functional as F

from chainer import reporter

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from e2e_asr_attctc_th import AttLoc


def encoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain('relu'))


def decoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain('tanh'))


class Reporter(chainer.Chain):
    def report(self, mse_loss, bce_loss, loss):
        reporter.report({'mse_loss': mse_loss}, self)
        reporter.report({'bce_loss': bce_loss}, self)
        reporter.report({'loss': loss}, self)


class Tacotron2(torch.nn.Module):
    """TCOTORON2 BASED SEQ2SEQ MODEL CONVERTS CHARS TO FEATURES

    :param int idim: dimension of the inputs
    :param int edim: dimension of character embedding
    :param int odim: dimension of target outputs
    :param int elayers: the number of encoder blstm layers
    :param int eunits: the number of encoder blstm units
    :param int dlayers: the number of decoder lstm layers
    :param int dunits: the number of decoder lstm units
    :param str atype: the name of attention type
    :param float threshold: threshold in inference
    :param float minlenratio: minimum length ratio
    :param float maxlenratio: maximum length ratio
    """

    def __init__(self, idim, edim, odim, elayers=1, eunits=512, dlayers=2, dunits=1024,
                 atype="location", adim=512, aconv_chans=10, aconv_filts=100, dropout=0.5,
                 threshold=0.5, minlenratio=0.0, maxlenratio=0.0):
        super(Tacotron2, self).__init__()
        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.elayers = elayers
        self.eunits = eunits
        self.dlayers = dlayers
        self.dunits = dunits
        self.atype = atype
        self.adim = adim
        self.aconv_filts = aconv_filts
        self.aconv_chans = aconv_chans
        self.dropout = dropout
        self.threshold = threshold
        self.minlenratio = minlenratio
        self.maxlenratio = maxlenratio

        # define network modules
        self.enc = Encoder(idim, edim, elayers, eunits, dropout)
        att = AttLoc(eunits, dunits, eunits, aconv_chans, aconv_filts)
        self.dec = Decoder(eunits, odim, att, dlayers, dunits, dropout)

        # initialize
        self.enc.apply(encoder_init)
        self.dec.apply(decoder_init)

        self.reporter = Reporter()

    def forward(self, xs, ilens, ys, olens):
        """TACOTRON2 FORWARD CALCULATION

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax)
        :param list olens: list of lengths of each target batch (B)
        :return: outputs with postnets (B, Lmax, D)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, D)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        :return: list of lengths of each batch (B)
        :rtype: list
        """
        hs, hlens = self.enc(xs, ilens)
        post_outs, outs, probs = self.dec(hs, hlens, ys, olens)
        return post_outs, outs, probs

    def loss(self, ys, post_outs, outs, probs, olens):
        """TACOTRON2 LOSS CALCULATION

        :param torch.Tensor ys: batch of padded targets (B, Lmax, D)
        :param torch.Tensor post_outs: outputs with postnets (B, Lmax, D)
        :param torch.Tensor outs: outputs without postnets (B, Lmax, D)
        :param torch.Tensor probs: stop logits (B, Lmax)
        :param list olens: list of lengths of each target batch (B)
        :return: mean squared error
        :rtype: torch.Tensor
        :return: binary cross entorpy
        :rtype: torch.Tensor
        """
        mse_loss = 0.0
        bce_loss = 0.0
        for y, post_out, out, prob, olen in zip(ys, post_outs, outs, probs, olens):
            mse_loss += torch.nn.functional.mse_loss(
                out[:olen], y[:olen], size_average=False)
            mse_loss += torch.nn.functional.mse_loss(
                post_out[:olen], y[:olen], size_average=False)
            target = torch.zeros(olen).float()
            target[-1] = 1.0
            if torch.cuda.is_available():
                target = target.cuda()
            bce_loss += torch.nn.functional.binary_cross_entropy(
                F.sigmoid(prob[:olen]), target, size_average=False)
        mse_loss /= sum(olens) * self.odim
        bce_loss /= sum(olens)
        loss = mse_loss + bce_loss

        self.reporter.report(mse_loss.item(), bce_loss.item(), loss.item())

        return loss

    def inference(self, x):
        """GENERATE THE SEQUENCE OF FEATURES

        :param tensor x: the sequence of characters (T)
        :return: the sequence of features (L, D)
        """
        # setup
        xs = x.unsqueeze(0)
        ilens = [x.size(0)]
        maxlen = int(x.size(0) * self.maxlenratio)
        minlen = int(x.size(0) * self.minlenratio)

        hs, hlens = self.enc(xs, ilens)

        # initialize hidden states of decoder
        c_list = [self.dec.zero_state(hs)]
        z_list = [self.dec.zero_state(hs)]
        for l in six.moves.range(1, self.dlayers):
            c_list += [self.dec.zero_state(hs)]
            z_list += [self.dec.zero_state(hs)]
        prev_out = hs.new_zeros(1, self.odim)

        # initialize attention
        att_w = None
        self.dec.att.reset()

        # loop for an output sequence
        idx = 0
        outs = []
        att_ws = []
        probs = []
        while True:
            # updated index
            idx += 1

            # decoder calculation
            att_c, att_w = self.dec.att(hs, hlens, z_list[0], att_w)
            att_ws += [att_w]
            prenet_out = self.dec.prenet(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.dec.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.dec.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            outs += [self.dec.feat_out(z_list[-1])]
            probs += [F.sigmoid(self.dec.prob_out(z_list[-1]))[0]]
            prev_out = outs[-1]

            # check whether to finish generation
            if (probs[-1] >= self.threshold and idx >= minlen) or idx == maxlen:
                outs = torch.stack(outs, dim=2)  # (1, odim, L)
                outs += self.dec.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (Lx, odim)
                return outs, probs, att_ws


class Encoder(torch.nn.Module):
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

    def __init__(self, idim, edim=512, elayers=1, eunits=512, dropout=0.5):
        super(Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.edim = edim
        self.elayers = elayers
        self.eunits = eunits
        self.dropout = dropout
        # define network layer modules
        self.embed = torch.nn.Embedding(idim, edim, padding_idx=0)
        self.convs = torch.nn.Sequential(
            torch.nn.Conv1d(edim, edim, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(edim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(edim, edim, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(edim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(edim, edim, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(edim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.blstm = torch.nn.LSTM(
            edim, eunits // 2, elayers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, xs, ilens):
        """CHARACTER ENCODER FORWARD CALCULATION

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each batch (B)

        :return: batch of sequences of padded encoder states (B, Tmax, eunits)
        :rtype: torch.Tensor
        :return: list of lengths of each batch (B)
        :rtype: list
        """
        out = self.embed(xs)  # (B, Tmax, C)
        out = self.convs(out.transpose(1, 2))  # (B, C, Tmax)
        out = pack_padded_sequence(out.transpose(1, 2), ilens, batch_first=True)
        out, _ = self.blstm(out)  # (B, Tmax, C)
        out, olens = pad_packed_sequence(out, batch_first=True)

        return out, olens.tolist()


class Decoder(torch.nn.Module):
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

    def __init__(self, idim, odim, att, dlayers=2, dunits=1024, dropout=0.5):
        super(Decoder, self).__init__()
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
            torch.nn.Linear(odim, 256, bias=False),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 256, bias=False),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.postnet = torch.nn.Sequential(
            torch.nn.Conv1d(odim, 512, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(512, 512, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(512, 512, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(512, 512, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(512, odim, 5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm1d(odim),
            torch.nn.Dropout(dropout),
        )
        # use 1x1 conv instead of linear
        self.feat_out = torch.nn.Linear(dunits, odim, bias=False)
        self.prob_out = torch.nn.Linear(dunits, 1)

    def zero_state(self, hs):
        return hs.data.new(hs.size(0), self.dunits).zero_()

    def forward(self, hs, hlens, ys, olens):
        """DECODER FORWARD CALCULATION

        :param torch.Tensor hs: batch of the sequences of padded hidden states (B, Tmax, idim)
        :param list hlens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of the sequences of padded target features (B, Lmax, odim)
        :param list ylens: list of lengths of each output batch (B)
        :return: outputs with postnets (B, Lmax, D)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, D)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        :return: list of lengths of each batch (B)
        :rtype: list
        """
        # initialize hidden states of decoder
        c_list = [self.zero_state(hs)]
        z_list = [self.zero_state(hs)]
        for l in six.moves.range(1, self.dlayers):
            c_list += [self.zero_state(hs)]
            z_list += [self.zero_state(hs)]
        prev_out = hs.new_zeros(hs.size(0), self.odim)

        # initialize attention
        att_w = None
        self.att.reset()

        # loop for an output sequence
        outs = []
        probs = []
        for i in six.moves.range(max(olens)):
            att_c, att_w = self.att(hs, hlens, z_list[0], att_w)
            prenet_out = self.prenet(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            outs += [self.feat_out(z_list[-1])]
            probs += [self.prob_out(z_list[-1])]
            prev_out = ys[:, i]  # teacher forcing

        outs = torch.stack(outs, dim=2)  # (B, odim, Lmax)
        probs = torch.cat(probs, dim=1)  # (B, Lmax)
        post_outs = outs + self.postnet(outs)  # (B, odim, Lmax)
        outs = outs.transpose(2, 1)  # (B, Lmax, odim)
        post_outs = post_outs.transpose(2, 1)  # (B, Lmax, odim)

        return post_outs, outs, probs
