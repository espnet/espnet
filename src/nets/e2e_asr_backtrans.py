#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import logging
import six

import chainer
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from e2e_asr_attctc_th import AttLoc


def encoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain('relu'))


def decoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain('tanh'))


def pad_list(batch, pad_value=float("nan")):
    """FUNCTION TO PAD VALUE

    :param list batch: list of the sequences [(T_1, D), (T_2, D), ..., (T_B, D)]
    :param float pad_value: value to pad
    :return: padded batch with the shape (B, Tmax, D)
    """
    bs = len(batch)
    if isinstance(batch[0], torch.Tensor):
        # for pytorch variable
        maxlen = max(b.size(0) for b in batch)
        batch_pad = batch[0].new_zeros(bs, maxlen, *batch[0].size()[1:]).fill_(pad_value)
        for i, b in enumerate(batch):
            batch_pad[i, :b.size(0)] = b
    else:
        # for numpy ndarray
        maxlen = max([b.shape[0] for b in batch])
        if len(batch[0].shape) >= 2:
            batch_pad = np.zeros((bs, maxlen) + batch[0].shape[1:])
        else:
            batch_pad = np.zeros((bs, maxlen))
        batch_pad.fill(pad_value)
        for i, b in enumerate(batch):
            batch_pad[i, :b.shape[0]] = b

    return batch_pad


def make_mask(lengths, dim=None):
    """FUNCTION TO MAKE BINARY MASK

    Args:
        length (list): list of lengths
        dim (int): # dimension

    Return:
        (torch.ByteTensor) binary mask tensor (B, Lmax, dim)
    """
    batch = len(lengths)
    maxlen = max(lengths)
    if dim is None:
        mask = torch.zeros(batch, maxlen)
    else:
        mask = torch.zeros(batch, maxlen, dim)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1

    if torch.cuda.is_available():
        return mask.byte().cuda()
    else:
        return mask.byte()


class Reporter(chainer.Chain):
    def report(self, mse_loss, bce_loss, loss):
        chainer.reporter.report({'mse_loss': mse_loss}, self)
        chainer.reporter.report({'bce_loss': bce_loss}, self)
        chainer.reporter.report({'loss': loss}, self)


class ZoneOutCell(torch.nn.Module):
    """ZONEOUT CELL

    This code is modified from https://github.com/eladhoffer/seq2seq.pytorch

    :param torch.nn.Module cell: pytorch recurrent cell
    :param float zoneout_prob: probability of zoneout
    """

    def __init__(self, cell, zoneout_prob=0.1):
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_prob = zoneout_prob
        if zoneout_prob > 1.0 or zoneout_prob < 0.0:
            raise ValueError("zoneout probability must be in the range from 0.0 to 1.0.")

    def forward(self, inputs, hidden):
        next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout_prob)
        return next_hidden

    def _zoneout(self, h, next_h, prob):
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple([self._zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)])

        if self.training:
            mask = h.new_zeros(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h


class Tacotron2Loss(torch.nn.Module):
    """TACOTRON2 LOSS FUNCTION

    :param torch.nn.Module model: tacotron2 model
    :param bool use_masking: whether to mask padded part in loss calculation
    :param float bce_pos_weight: weight of positive sample of stop token (only for use_masking=True)
    """

    def __init__(self, model, use_masking=True, bce_pos_weight=20.0):
        super(Tacotron2Loss, self).__init__()
        self.model = model
        self.use_masking = use_masking
        self.bce_pos_weight = bce_pos_weight
        self.reporter = Reporter()

    def forward(self, xs, ilens, ys, labels, olens=None):
        """TACOTRON2 LOSS FORWARD CALCULATION

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor labels: batch of the sequences of stop token labels (B, Lmax)
        :param list olens: batch of the lengths of each target (B)
        :return: loss value
        :rtype: torch.Tensor
        """
        after_outs, before_outs, logits = self.model(xs, ilens, ys)
        if self.use_masking and olens is not None:
            # weight positive samples
            if self.bce_pos_weight != 1.0:
                weights = ys.new(*labels.size()).fill_(1)
                weights.masked_fill_(labels.eq(1), self.bce_pos_weight)
            else:
                weights = None
            # masking padded values
            mask = make_mask(olens, ys.size(2))
            ys = ys.masked_select(mask)
            after_outs = after_outs.masked_select(mask)
            before_outs = before_outs.masked_select(mask)
            labels = labels.masked_select(mask[:, :, 0])
            logits = logits.masked_select(mask[:, :, 0])
            weights = weights.masked_select(mask[:, :, 0]) if weights is not None else None
            # calculate loss
            mse_loss = F.mse_loss(after_outs, ys) + F.mse_loss(before_outs, ys)
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels, weights)
            loss = mse_loss + bce_loss
        else:
            # calculate loss
            mse_loss = F.mse_loss(after_outs, ys) + F.mse_loss(before_outs, ys)
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = mse_loss + bce_loss

        logging.debug("loss = %.3e (bce: %.3e, mse: %.3e)" % (loss.item(), bce_loss.item(), mse_loss.item()))
        self.reporter.report(mse_loss.item(), bce_loss.item(), loss.item())

        return loss


class Tacotron2(torch.nn.Module):
    """TACOTRON2 BASED SEQ2SEQ MODEL CONVERTS CHARS TO FEATURES

    :param int idim: dimension of the inputs
    :param int odim: dimension of the outputs
    :param int embed_dim: dimension of character embedding
    :param int elayers: the number of encoder blstm layers
    :param int eunits: the number of encoder blstm units
    :param int econv_layers: the number of encoder conv layers
    :param int econv_filts: the number of encoder conv filter size
    :param int econv_chans: the number of encoder conv filter channels
    :param int dlayers: the number of decoder lstm layers
    :param int dunits: the number of decoder lstm units
    :param int prenet_layers: the number of prenet layers
    :param int prenet_units: the number of prenet units
    :param int postnet_layers: the number of postnet layers
    :param int postnet_filts: the number of postnet filter size
    :param int postnet_chans: the number of postnet filter channels
    :param function output_activation_fn: activation function for outputs
    :param int adim: the number of dimension of mlp in attention
    :param int aconv_chans: the number of attention conv filter channels
    :param int aconv_filts: the number of attention conv filter size
    :param bool cumulate_att_w: whether to cumulate previous attention weight
    :param bool use_batch_norm: whether to use batch normalization
    :param bool use_concate: whether to concatenate encoder embedding with decoder lstm outputs
    :param float dropout: dropout rate
    :param float zoneout: zoneout rate
    :param float threshold: threshold in inference
    :param float minlenratio: minimum length ratio in inference
    :param float maxlenratio: maximum length ratio in inference
    """

    def __init__(self, idim, odim,
                 embed_dim=512,
                 elayers=1,
                 eunits=512,
                 econv_layers=3,
                 econv_filts=5,
                 econv_chans=512,
                 dlayers=2,
                 dunits=1024,
                 prenet_layers=2,
                 prenet_units=256,
                 postnet_layers=5,
                 postnet_filts=5,
                 postnet_chans=512,
                 output_activation_fn=None,
                 adim=512,
                 aconv_chans=32,
                 aconv_filts=15,
                 cumulate_att_w=True,
                 use_batch_norm=True,
                 use_concate=True,
                 dropout=0.5,
                 zoneout=0.1,
                 threshold=0.5,
                 maxlenratio=5.0,
                 minlenratio=0.0):
        super(Tacotron2, self).__init__()
        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.embed_dim = embed_dim
        self.elayers = elayers
        self.eunits = eunits
        self.econv_layers = econv_layers
        self.econv_filts = econv_filts
        self.econv_chans = econv_chans
        self.dlayers = dlayers
        self.dunits = dunits
        self.prenet_layers = prenet_layers
        self.prenet_units = prenet_units
        self.postnet_layers = postnet_layers
        self.postnet_chans = postnet_chans
        self.postnet_filts = postnet_filts
        self.output_activation_fn = output_activation_fn
        self.adim = adim
        self.aconv_filts = aconv_filts
        self.aconv_chans = aconv_chans
        self.cumulate_att_w = cumulate_att_w
        self.use_batch_norm = use_batch_norm
        self.use_concate = use_concate
        self.dropout = dropout
        self.zoneout = zoneout
        self.threshold = threshold
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        # define network modules
        self.enc = Encoder(idim=self.idim,
                           embed_dim=self.embed_dim,
                           elayers=self.elayers,
                           eunits=self.eunits,
                           econv_layers=self.econv_layers,
                           econv_chans=self.econv_chans,
                           econv_filts=self.econv_filts,
                           use_batch_norm=self.use_batch_norm,
                           dropout=self.dropout)
        self.dec = Decoder(idim=self.eunits,
                           odim=self.odim,
                           att=AttLoc(
                               self.eunits,
                               self.dunits,
                               self.adim,
                               self.aconv_chans,
                               self.aconv_filts),
                           dlayers=self.dlayers,
                           dunits=self.dunits,
                           prenet_layers=self.prenet_layers,
                           prenet_units=self.prenet_units,
                           postnet_layers=self.postnet_layers,
                           postnet_chans=self.postnet_chans,
                           postnet_filts=self.postnet_filts,
                           output_activation_fn=self.output_activation_fn,
                           cumulate_att_w=self.cumulate_att_w,
                           use_batch_norm=self.use_batch_norm,
                           use_concate=self.use_concate,
                           dropout=self.dropout,
                           zoneout=self.zoneout,
                           threshold=self.threshold,
                           maxlenratio=self.maxlenratio,
                           minlenratio=self.minlenratio)
        # initialize
        self.enc.apply(encoder_init)
        self.dec.apply(decoder_init)

    def forward(self, xs, ilens, ys):
        """TACOTRON2 FORWARD CALCULATION

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :return: outputs with postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        :return: attetion weights (B, Lmax, Tmax)
        :rtype: torch.Tensor
        """
        # check ilens type (should be list of int)
        if isinstance(ilens, torch.Tensor) or isinstance(ilens, np.ndarray):
            ilens = list(map(int, ilens))

        hs, hlens = self.enc(xs, ilens)
        after_outs, before_outs, logits = self.dec(hs, hlens, ys)

        return after_outs, before_outs, logits

    def inference(self, x):
        """GENERATE THE SEQUENCE OF FEATURES FROM THE SEQUENCE OF CHARACTERS

        :param tensor x: the sequence of characters (T)
        :return: the sequence of features (L, odim)
        :rtype: tensor
        :return: the sequence of stop probabilities (L)
        :rtype: tensor
        :return: the sequence of attetion weight (L, T)
        :rtype: tensor
        """
        h = self.enc.inference(x)
        outs, probs, att_ws = self.dec.inference(h)

        return outs, probs, att_ws

    def calculate_all_attentions(self, xs, ilens, ys):
        """TACOTRON2 FORWARD CALCULATION

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :return: attetion weights (B, Lmax, Tmax)
        :rtype: numpy array
        """
        # check ilens type (should be list of int)
        if isinstance(ilens, torch.Tensor) or isinstance(ilens, np.ndarray):
            ilens = list(map(int, ilens))

        with torch.no_grad():
            self.eval()
            hs, hlens = self.enc(xs, ilens)
            att_ws = self.dec.calculate_all_attentions(hs, hlens, ys)
            self.train()

        return att_ws.cpu().numpy()


class Encoder(torch.nn.Module):
    """CHARACTER EMBEDDING ENCODER

    This is the encoder which converts the sequence of characters into
    the sequence of hidden states. The newtwork structure is based on
    that of tacotron2 in the field of speech synthesis.

    :param int idim: dimension of the inputs
    :param int embed_dim: dimension of character embedding
    :param int elayers: the number of encoder blstm layers
    :param int eunits: the number of encoder blstm units
    :param int econv_layers: the number of encoder conv layers
    :param int econv_filts: the number of encoder conv filter size
    :param int econv_chans: the number of encoder conv filter channels
    :param bool use_batch_norm: whether to use batch normalization
    :param float dropout: dropout rate
    """

    def __init__(self, idim,
                 embed_dim=512,
                 elayers=1,
                 eunits=512,
                 econv_layers=3,
                 econv_chans=512,
                 econv_filts=5,
                 use_batch_norm=True,
                 use_residual=False,
                 dropout=0.5):
        super(Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.embed_dim = embed_dim
        self.elayers = elayers
        self.eunits = eunits
        self.econv_layers = econv_layers
        self.econv_chans = econv_chans if econv_layers != 0 else -1
        self.econv_filts = econv_filts if econv_layers != 0 else -1
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.dropout = dropout
        # define network layer modules
        self.embed = torch.nn.Embedding(self.idim, self.embed_dim)
        if self.econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in six.moves.range(self.econv_layers):
                ichans = self.embed_dim if layer == 0 else self.econv_chans
                if self.use_batch_norm:
                    self.convs += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, self.econv_chans, self.econv_filts, stride=1,
                                        padding=(self.econv_filts - 1) // 2, bias=False),
                        torch.nn.BatchNorm1d(self.econv_chans),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(self.dropout))]
                else:
                    self.convs += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, self.econv_chans, self.econv_filts, stride=1,
                                        padding=(self.econv_filts - 1) // 2, bias=False),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(self.dropout))]
        else:
            self.convs = None
        iunits = econv_chans if self.econv_layers != 0 else self.embed_dim
        self.blstm = torch.nn.LSTM(
            iunits, self.eunits // 2, self.elayers,
            batch_first=True,
            bidirectional=True)

    def forward(self, xs, ilens):
        """CHARACTER ENCODER FORWARD CALCULATION

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each batch (B)
        :return: batch of sequences of padded encoder states (B, Tmax, eunits)
        :rtype: torch.Tensor
        :return: batch of lenghts of each encoder states (B)
        :rtype: list
        """
        xs = self.embed(xs).transpose(1, 2)
        for l in six.moves.range(self.econv_layers):
            if self.use_residual:
                xs += self.convs[l](xs)
            else:
                xs = self.convs[l](xs)
        xs = pack_padded_sequence(xs.transpose(1, 2), ilens, batch_first=True)
        # does not work with dataparallel
        # see https://github.com/pytorch/pytorch/issues/7092#issuecomment-388194623
        # self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Tmax, C)
        xs, hlens = pad_packed_sequence(xs, batch_first=True)

        return xs, list(map(int, hlens))

    def inference(self, x):
        """CHARACTER ENCODER INFERENCE

        :param torch.Tensor x: the sequence of character ids (T)
        :return: the sequence encoder states (T, eunits)
        :rtype: torch.Tensor
        """
        assert len(x.size()) == 1
        xs = x.unsqueeze(0)
        ilens = [x.size(0)]

        return self.forward(xs, ilens)[0][0]


class Decoder(torch.nn.Module):
    """DECODER TO PREDICT THE SEQUENCE OF FEATURES

    This the decoder which generate the sequence of features from
    the sequence of the hidden states. The network structure is
    based on that of the tacotron2 in the field of speech synthesis.

    :param int idim: dimension of the inputs
    :param int odim: dimension of the outputs
    :param instance att: instance of attetion class
    :param int dlayers: the number of decoder lstm layers
    :param int dunits: the number of decoder lstm units
    :param int prenet_layers: the number of prenet layers
    :param int prenet_units: the number of prenet units
    :param int postnet_layers: the number of postnet layers
    :param int postnet_filts: the number of postnet filter size
    :param int postnet_chans: the number of postnet filter channels
    :param function output_activation_fn: activation function for outputs
    :param bool cumulate_att_w: whether to cumulate previous attention weight
    :param bool use_batch_norm: whether to use batch normalization
    :param bool use_concate: whether to concatenate encoder embedding with decoder lstm outputs
    :param float dropout: dropout rate
    :param float zoneout: zoneout rate
    :param float threshold: threshold in inference
    :param float minlenratio: minimum length ratio in inference
    :param float maxlenratio: maximum length ratio in inference
    """

    def __init__(self, idim, odim, att,
                 dlayers=2,
                 dunits=1024,
                 prenet_layers=2,
                 prenet_units=256,
                 postnet_layers=5,
                 postnet_chans=512,
                 postnet_filts=5,
                 output_activation_fn=None,
                 cumulate_att_w=True,
                 use_batch_norm=True,
                 use_concate=True,
                 dropout=0.5,
                 zoneout=0.1,
                 threshold=0.5,
                 maxlenratio=5.0,
                 minlenratio=0.0):
        super(Decoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.odim = odim
        self.att = att
        self.dlayers = dlayers
        self.dunits = dunits
        self.prenet_layers = prenet_layers
        self.prenet_units = prenet_units if prenet_layers != 0 else self.odim
        self.postnet_layers = postnet_layers
        self.postnet_chans = postnet_chans if postnet_layers != 0 else -1
        self.postnet_filts = postnet_filts if postnet_layers != 0 else -1
        self.output_activation_fn = output_activation_fn
        self.cumulate_att_w = cumulate_att_w
        self.use_batch_norm = use_batch_norm
        self.use_concate = use_concate
        self.dropout = dropout
        self.zoneout = zoneout
        self.threshold = threshold
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        # define lstm network
        self.lstm = torch.nn.ModuleList()
        for layer in six.moves.range(self.dlayers):
            iunits = self.idim + self.prenet_units if layer == 0 else self.dunits
            lstm = torch.nn.LSTMCell(iunits, self.dunits)
            if zoneout > 0.0:
                lstm = ZoneOutCell(lstm, self.zoneout)
            self.lstm += [lstm]
        # define prenet
        if self.prenet_layers > 0:
            self.prenet = torch.nn.ModuleList()
            for layer in six.moves.range(self.prenet_layers):
                ichans = self.odim if layer == 0 else self.prenet_units
                self.prenet += [torch.nn.Sequential(
                    torch.nn.Linear(ichans, self.prenet_units, bias=False),
                    torch.nn.ReLU())]
        else:
            self.prenet = None
        # define postnet
        if self.postnet_layers > 0:
            self.postnet = torch.nn.ModuleList()
            for layer in six.moves.range(self.postnet_layers - 1):
                ichans = self.odim if layer == 0 else self.postnet_chans
                ochans = self.odim if layer == self.postnet_layers - 1 else self.postnet_chans
                if use_batch_norm:
                    self.postnet += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, ochans, self.postnet_filts, stride=1,
                                        padding=(self.postnet_filts - 1) // 2, bias=False),
                        torch.nn.BatchNorm1d(ochans),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(self.dropout))]
                else:
                    self.postnet += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, ochans, self.postnet_filts, stride=1,
                                        padding=(self.postnet_filts - 1) // 2, bias=False),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(self.dropout))]
            ichans = self.postnet_chans if self.postnet_layers != 1 else self.odim
            if use_batch_norm:
                self.postnet += [torch.nn.Sequential(
                    torch.nn.Conv1d(ichans, odim, self.postnet_filts, stride=1,
                                    padding=(self.postnet_filts - 1) // 2, bias=False),
                    torch.nn.BatchNorm1d(odim),
                    torch.nn.Dropout(self.dropout))]
            else:
                self.postnet += [torch.nn.Sequential(
                    torch.nn.Conv1d(ichans, odim, self.postnet_filts, stride=1,
                                    padding=(self.postnet_filts - 1) // 2, bias=False),
                    torch.nn.Dropout(self.dropout))]
        else:
            self.postnet = None
        # define projection layers
        iunits = self.idim + self.dunits if self.use_concate else self.dunits
        self.feat_out = torch.nn.Linear(iunits, self.odim, bias=False)
        self.prob_out = torch.nn.Linear(iunits, 1)

    def zero_state(self, hs):
        return hs.new(hs.size(0), self.dunits).zero_()

    def forward(self, hs, hlens, ys, att_w_maxlen=None):
        """DECODER FORWARD CALCULATION

        :param torch.Tensor hs: batch of the sequences of padded hidden states (B, Tmax, idim)
        :param list hlens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of the sequences of padded target features (B, Lmax, odim)
        :param int att_w_maxlen: maximum length of att_w (only for dataparallel)
        :return: outputs with postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        :return: attetion weights (B, Lmax, Tmax)
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
            prenet_out = self._prenet_forward(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            zcs = torch.cat([z_list[-1], att_c], dim=1) if self.use_concate else z_list[-1]
            outs += [self.feat_out(zcs)]
            logits += [self.prob_out(zcs)]
            prev_out = y  # teacher forcing
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w

        logits = torch.cat(logits, dim=1)  # (B, Lmax)
        before_outs = torch.stack(outs, dim=2)  # (B, odim, Lmax)
        after_outs = before_outs + self._postnet_forward(before_outs)  # (B, odim, Lmax)
        before_outs = before_outs.transpose(2, 1)  # (B, Lmax, odim)
        after_outs = after_outs.transpose(2, 1)  # (B, Lmax, odim)

        # apply activation function for scaling
        if self.output_activation_fn is not None:
            before_outs = self.output_activation_fn(before_outs)
            after_outs = self.output_activation_fn(after_outs)

        return after_outs, before_outs, logits

    def inference(self, h):
        """GENERATE THE SEQUENCE OF FEATURES FROM ENCODER HIDDEN STATES

        :param tensor h: the sequence of encoder states (T, C)
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
        maxlen = int(h.size(0) * self.maxlenratio)
        minlen = int(h.size(0) * self.minlenratio)

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
            prenet_out = self._prenet_forward(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            zcs = torch.cat([z_list[-1], att_c], dim=1) if self.use_concate else z_list[-1]
            if self.output_activation_fn is not None:
                outs += [self.output_activation_fn(self.feat_out(zcs))]
            else:
                outs += [self.feat_out(zcs)]
            probs += [F.sigmoid(self.prob_out(zcs))[0]]
            prev_out = outs[-1]
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w

            # check whether to finish generation
            if (probs[-1] >= self.threshold and idx >= minlen) or idx == maxlen:
                outs = torch.stack(outs, dim=2)  # (1, odim, L)
                outs = outs + self._postnet_forward(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (Lx, odim)
                probs = torch.cat(probs, dim=0)
                att_ws = torch.cat(att_ws, dim=0)
                break

        return outs, probs, att_ws

    def calculate_all_attentions(self, hs, hlens, ys):
        """DECODER ATTENTION CALCULATION

        :param torch.Tensor hs: batch of the sequences of padded hidden states (B, Tmax, idim)
        :param list hlens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of the sequences of padded target features (B, Lmax, odim)
        :return: attetion weights (B, Lmax, Tmax)
        :rtype: numpy array
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
        att_ws = []
        for y in ys.transpose(0, 1):
            att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w)
            att_ws += [att_w]
            prenet_out = self._prenet_forward(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            prev_out = y  # teacher forcing
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w

        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)

        return att_ws

    def _prenet_forward(self, x):
        if self.prenet is not None:
            for l in six.moves.range(self.prenet_layers):
                x = F.dropout(self.prenet[l](x), self.dropout)
        return x

    def _postnet_forward(self, xs):
        if self.postnet is not None:
            for l in six.moves.range(self.postnet_layers):
                xs = self.postnet[l](xs)
        return xs
