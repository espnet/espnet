#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import logging
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


def make_mask(lengths, dim):
    """FUNCTION TO MAKE BINARY MASK

    Args:
        length (list): list of lengths
        dim (int): # dimension

    Return:
        (torch.ByteTensor) binary mask tensor (B, Lmax, dim)
    """
    batch = len(lengths)
    maxlen = max(lengths)
    mask = torch.zeros(batch, maxlen, dim)
    for i, l in enumerate(lengths):
        mask[i, l:] = 1

    return mask.byte()


class Tacotron2(torch.nn.Module):
    """TCOTORON2 BASED SEQ2SEQ MODEL CONVERTS CHARS TO FEATURES

    :param int idim: dimension of the inputs
    :param int edim: dimension of character embedding
    :param int odim: dimension of target outputs
    :param int elayers: the number of encoder blstm layers
    :param int eunits: the number of encoder blstm units
    :param int dlayers: the number of decoder lstm layers
    :param int dunits: the number of decoder lstm units
    """

    def __init__(self, idim, edim, odim,
                 elayers=1, eunits=512, dlayers=2, dunits=1024,
                 adim=512, achans=10, afilts=100, cumulate_att_w=True,
                 dropout=0.5):
        super(Tacotron2, self).__init__()
        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.elayers = elayers
        self.eunits = eunits
        self.dlayers = dlayers
        self.dunits = dunits
        self.adim = adim
        self.afilts = afilts
        self.achans = achans
        self.cumulate_att_w = cumulate_att_w
        self.dropout = dropout

        # define network modules
        self.enc = Encoder(idim, edim, elayers, eunits, dropout)
        att = AttLoc(eunits, dunits, eunits, achans, afilts)
        self.dec = Decoder(eunits, odim, att, dlayers, dunits, dropout, cumulate_att_w)

        # initialize
        self.enc.apply(encoder_init)
        self.dec.apply(decoder_init)

        self.reporter = Reporter()

    def forward(self, xs, ilens, ys):
        """TACOTRON2 FORWARD CALCULATION

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax)
        :return: outputs with postnets (B, Lmax, D)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, D)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        """
        hs = self.enc(xs, ilens)
        after_outs, before_outs, logits = self.dec(hs, ilens, ys)

        return after_outs, before_outs, logits

    def inference(self, x, threshold=0.5, minlenratio=0.0, maxlenratio=100.0):
        """GENERATE THE SEQUENCE OF FEATURES FROM THE SEQUENCE OF CHARACTERS

        :param tensor x: the sequence of characters (T)
        :param float threshold: threshold in inference
        :param float minlenratio: minimum length ratio
        :param float maxlenratio: maximum length ratio
        :return: the sequence of features (L, D)
        :rtype: tensor
        :return: the sequence of stop probabilities (L)
        :rtype: tensor
        :return: the sequence of attetion weight (L, T)
        :rtype: tensor
        """
        h = self.enc.inference(x)
        outs, probs, att_ws = self.dec.inference(h, threshold, minlenratio, maxlenratio)

        return outs, probs, att_ws

    def loss(self, targets, outputs, olens=None, masking=False, bce_pos_weight=1.0):
        """TACOTRON2 LOSS CALCULATION

        :param tuple targets: consist of (ys, lables), where
            ys is target padded sequence (B, Lmax, D), and
            labels is target stop token sequence (B, Lmax)
        :param tuple outputs: consist of (after_outs, before_outs, logits), where
            after_outs is Tacotron2 outputs after postnets (B, Lmax, D),
            before_outs is Tacotron2 outputs before postnets (B, Lmax, D), and
            logits is Tacotron2 stop token sequence (B, Lmax)
        :param list olens: list of lengths of each target batch (B)
        :param bool masking: whether to mask padded part
        :param float bce_pos_weight: weight of positive sample of stop token
            if masking = False, will not effect at all
        :return: tacotron2 loss
        :rtype: torch.Tensor
        """
        # parse targets and outputs
        ys, labels = targets
        after_outs, before_outs, logits = outputs

        # masking padded part
        if masking and olens is not None:
            if bce_pos_weight != 1.0:
                weights = ys.new(ys.size(0)).fill_(1)
                weights.masked_fill_(ys.eq(1), bce_pos_weight)
            mask = make_mask(olens, ys.size(2))
            after_outs = after_outs.masked_fill_(mask, 0)
            before_outs = before_outs.masked_fill_(mask, 0)
            logits = logits.masked_fill_(mask[:, :, 0], 1e3)
        else:
            weights = None

        # calculate loss
        mse_loss = F.mse_loss(after_outs, ys) + F.mse_loss(before_outs, ys)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss = mse_loss + bce_loss

        # report
        logging.info("mse loss = %.5e" % mse_loss.item())
        logging.info("bce loss = %.5e" % bce_loss.item())
        self.reporter.report(mse_loss.item(), bce_loss.item(), loss.item())

        return loss


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
        self.embed = torch.nn.Embedding(idim, edim)
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
        """
        outs = self.embed(xs)  # (B, Tmax, C)
        outs = self.convs(outs.transpose(1, 2))  # (B, C, Tmax)
        outs = pack_padded_sequence(outs.transpose(1, 2), ilens, batch_first=True)
        self.blstm.flatten_parameters()
        outs, _ = self.blstm(outs)  # (B, Tmax, C)
        outs, _ = pad_packed_sequence(outs, batch_first=True)

        return outs

    def inference(self, x):
        """CHARACTER ENCODER INFERENCE

        :param torch.Tensor x: the sequence of character ids (T)
        :return: the sequence encoder states (T, eunits)
        :rtype: torch.Tensor
        """
        assert len(x.size()) == 1
        xs = x.unsqueeze(0)
        ilens = [x.size(0)]

        return self.forward(xs, ilens)[0]


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

    def __init__(self, idim, odim, att, dlayers=2, dunits=1024, dropout=0.5, cumulate_att_w=True):
        super(Decoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.odim = odim
        self.att = att
        self.dlayers = dlayers
        self.dunits = dunits
        self.cumulate_att_w = cumulate_att_w
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
        self.feat_out = torch.nn.Linear(idim + dunits, odim, bias=False)
        self.prob_out = torch.nn.Linear(idim + dunits, 1)

    def zero_state(self, hs):
        return hs.data.new(hs.size(0), self.dunits).zero_()

    def forward(self, hs, hlens, ys):
        """DECODER FORWARD CALCULATION

        :param torch.Tensor hs: batch of the sequences of padded hidden states (B, Tmax, idim)
        :param list hlens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of the sequences of padded target features (B, Lmax, odim)
        :return: outputs with postnets (B, Lmax, D)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, D)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        """
        # initialize hidden states of decoder
        c_list = [self.zero_state(hs)]
        z_list = [self.zero_state(hs)]
        for l in six.moves.range(1, self.dlayers):
            c_list += [self.zero_state(hs)]
            z_list += [self.zero_state(hs)]
        prev_out = hs.new_zeros(hs.size(0), self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # loop for an output sequence
        outs, logits = [], []
        for y in ys.transpose(0, 1):
            att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w)
            prenet_out = self.prenet(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            zcs = torch.cat([z_list[-1], att_c], dim=1)
            outs += [self.feat_out(zcs)]
            logits += [self.prob_out(zcs)]
            prev_out = y  # teacher forcing
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w

        logits = torch.cat(logits, dim=1)  # (B, Lmax)
        before_outs = torch.stack(outs, dim=2)  # (B, odim, Lmax)
        after_outs = before_outs + self.postnet(before_outs)  # (B, odim, Lmax)
        before_outs = before_outs.transpose(2, 1)  # (B, Lmax, odim)
        after_outs = after_outs.transpose(2, 1)  # (B, Lmax, odim)

        return after_outs, before_outs, logits

    def inference(self, h, threshold=0.5, minlenratio=0.0, maxlenratio=100.0):
        """GENERATE THE SEQUENCE OF FEATURES FROM ENCODER HIDDEN STATES

        :param tensor h: the sequence of encoder states (T, C)
        :param float threshold: threshold in inference
        :param float minlenratio: minimum length ratio
        :param float maxlenratio: maximum length ratio
        :return: the sequence of features (L, D)
        :rtype: tensor
        :return: the sequence of stop probabilities (L)
        :rtype: tensor
        :return: the sequence of attetion weight (L, T)
        :rtype: tensor
        """
        # setup
        assert len(h.size()) == 2
        hs = h.unsqueeze(0)
        ilens = [h.size(0)]
        maxlen = int(h.size(0) * maxlenratio)
        minlen = int(h.size(0) * minlenratio)

        # initialize hidden states of decoder
        c_list = [self.zero_state(hs)]
        z_list = [self.zero_state(hs)]
        for l in six.moves.range(1, self.dlayers):
            c_list += [self.zero_state(hs)]
            z_list += [self.zero_state(hs)]
        prev_out = hs.new_zeros(1, self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # loop for an output sequence
        idx = 0
        outs, att_ws, probs = [], [], []
        while True:
            # updated index
            idx += 1

            # decoder calculation
            att_c, att_w = self.att(hs, ilens, z_list[0], prev_att_w)
            att_ws += [att_w]
            prenet_out = self.prenet(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            zcs = torch.cat([z_list[-1], att_c], dim=1)
            outs += [self.feat_out(zcs)]
            probs += [F.sigmoid(self.prob_out(zcs))[0]]
            prev_out = outs[-1]
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w

            # check whether to finish generation
            if (probs[-1] >= threshold and idx >= minlen) or idx == maxlen:
                outs = torch.stack(outs, dim=2)  # (1, odim, L)
                outs += self.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (Lx, odim)
                probs = torch.cat(probs, dim=0)
                att_ws = torch.cat(att_ws, dim=0)
                break

        return outs, probs, att_ws
