#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging
import math
import sys

from argparse import Namespace

import chainer
import numpy as np
import six
import torch
import torch.nn.functional as F
import warpctc_pytorch as warp_ctc

from chainer import reporter
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from ctc_prefix_score import CTCPrefixScore
from e2e_asr_common import end_detect
from e2e_asr_common import get_vgg2l_odim
from e2e_asr_common import label_smoothing_dist


CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5


# ------------- Utility functions --------------------------------------------------------------------------------------
def to_cuda(m, x):
    """Function to send tensor into corresponding device

    :param torch.nn.Module m: torch module
    :param torch.Tensor x: torch tensor
    :return: torch tensor located in the same place as torch module
    :rtype: torch.Tensor
    """
    assert isinstance(m, torch.nn.Module)
    device = next(m.parameters()).device
    return x.to(device)


def pad_list(xs, pad_value):
    """Function to pad values

    :param list xs: list of torch.Tensor [(L_1, D), (L_2, D), ..., (L_B, D)]
    :param float pad_value: value for padding
    :return: padded tensor (B, Lmax, D)
    :rtype: torch.Tensor
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def make_pad_mask(lengths):
    """Function to make mask tensor containing indices of padded part

    e.g.: lengths = [5, 3, 2]
          mask = [[0, 0, 0, 0 ,0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 1]]

    :param list lengths: list of lengths (B)
    :return: mask tensor containing indices of padded part (B, Tmax)
    :rtype: torch.Tensor
    """
    bs = int(len(lengths))
    maxlen = int(max(lengths))
    mask = torch.zeros(bs, maxlen, dtype=torch.uint8)
    for i, l in enumerate(lengths):
        mask[i, l:] = 1

    return mask


def th_accuracy(pad_outputs, pad_targets, ignore_label):
    """Function to calculate accuracy

    :param torch.Tensor pad_outputs: prediction tensors (B*Lmax, D)
    :param torch.Tensor pad_targets: target tensors (B, Lmax, D)
    :param int ignore_label: ignore label id
    :retrun: accuracy value (0.0 - 1.0)
    :rtype: float
    """
    pad_pred = pad_outputs.view(
        pad_targets.size(0),
        pad_targets.size(1),
        pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


class Reporter(chainer.Chain):
    def report(self, loss_ctc, loss_att, acc, mtl_loss):
        reporter.report({'loss_ctc': loss_ctc}, self)
        reporter.report({'loss_att': loss_att}, self)
        reporter.report({'acc': acc}, self)
        logging.info('mtl loss:' + str(mtl_loss))
        reporter.report({'loss': mtl_loss}, self)


# TODO(watanabe) merge Loss and E2E: there is no need to make these separately
class Loss(torch.nn.Module):
    """Multi-task learning loss module

    :param torch.nn.Module predictor: E2E model instance
    :param float mtlalpha: mtl coefficient value (0.0 ~ 1.0)
    """

    def __init__(self, predictor, mtlalpha):
        super(Loss, self).__init__()
        assert 0.0 <= mtlalpha <= 1.0, "mtlalpha shoule be [0.0, 1.0]"
        self.mtlalpha = mtlalpha
        self.loss = None
        self.accuracy = None
        self.predictor = predictor
        self.reporter = Reporter()

    def forward(self, xs_pad, ilens, ys_pad):
        '''Multi-task learning loss forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        '''
        self.loss = None
        loss_ctc, loss_att, acc = self.predictor(xs_pad, ilens, ys_pad)
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data, loss_att_data, acc, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        return self.loss


class E2E(torch.nn.Module):
    """E2E module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param namespace args: argument namespace containing options
    """

    def __init__(self, idim, odim, args):
        super(E2E, self).__init__()
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.mtlalpha = args.mtlalpha

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers + 1, dtype=np.int)
        if args.etype == 'blstmp':
            ss = args.subsample.split("_")
            for j in range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args.lsm_type:
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # encoder
        self.enc = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs,
                           self.subsample, args.dropout_rate)
        # ctc
        self.ctc = CTC(odim, args.eprojs, args.dropout_rate)
        # attention
        if args.atype == 'noatt':
            self.att = NoAtt()
        elif args.atype == 'dot':
            self.att = AttDot(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'add':
            self.att = AttAdd(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'location':
            self.att = AttLoc(args.eprojs, args.dunits,
                              args.adim, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'location2d':
            self.att = AttLoc2D(args.eprojs, args.dunits,
                                args.adim, args.awin, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'location_recurrent':
            self.att = AttLocRec(args.eprojs, args.dunits,
                                 args.adim, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'coverage':
            self.att = AttCov(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'coverage_location':
            self.att = AttCovLoc(args.eprojs, args.dunits, args.adim,
                                 args.aconv_chans, args.aconv_filts)
        elif args.atype == 'multi_head_dot':
            self.att = AttMultiHeadDot(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim)
        elif args.atype == 'multi_head_add':
            self.att = AttMultiHeadAdd(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim)
        elif args.atype == 'multi_head_loc':
            self.att = AttMultiHeadLoc(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim,
                                       args.aconv_chans, args.aconv_filts)
        elif args.atype == 'multi_head_multi_res_loc':
            self.att = AttMultiHeadMultiResLoc(args.eprojs, args.dunits,
                                               args.aheads, args.adim, args.adim,
                                               args.aconv_chans, args.aconv_filts)
        else:
            logging.error(
                "Error: need to specify an appropriate attention archtecture")
            sys.exit()
        # decoder
        self.dec = Decoder(args.eprojs, odim, args.dlayers, args.dunits,
                           self.sos, self.eos, self.att, self.verbose, self.char_list,
                           labeldist, args.lsm_weight)

        # weight initialization
        self.init_like_chainer()

    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        def lecun_normal_init_parameters(module):
            for p in module.parameters():
                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 4:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

    def forward(self, xs_pad, ilens, ys_pad):
        '''E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        '''
        # 1. encoder
        hs_pad, hlens = self.enc(xs_pad, ilens)

        # 2. CTC loss
        if self.mtlalpha == 0:
            loss_ctc = None
        else:
            loss_ctc = self.ctc(hs_pad, hlens, ys_pad)

        # 3. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
        else:
            loss_att, acc = self.dec(hs_pad, hlens, ys_pad)

        return loss_ctc, loss_att, acc

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        '''E2E beam search

        :param ndarray x: input acouctic feature (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        '''
        prev = self.training
        self.eval()
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_cuda(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))

        # 1. encoder
        # make a utt list (1) to use the same interface for encoder
        h, _ = self.enc(h.unsqueeze(0), ilen)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(h)[0]
        else:
            lpz = None

        # 2. decoder
        # decode the first utterance
        y = self.dec.recognize_beam(h[0], lpz, recog_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        '''E2E attention calculation

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        '''
        with torch.no_grad():
            # encoder
            hpad, hlens = self.enc(xs_pad, ilens)

            # decoder
            att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys_pad)

        return att_ws


# ------------- CTC Network --------------------------------------------------------------------------------------------
class CTC(torch.nn.Module):
    """CTC module

    :param int odim: dimension of outputs
    :param int eprojs: number of encoder projection units
    :param float dropout_rate: dropout rate (0.0 ~ 1.0)
    """

    def __init__(self, odim, eprojs, dropout_rate):
        super(CTC, self).__init__()
        self.dropout_rate = dropout_rate
        self.loss = None
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.loss_fn = warp_ctc.CTCLoss(size_average=True)
        self.ignore_id = -1

    def forward(self, hs_pad, hlens, ys_pad):
        '''CTC forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        '''
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

        self.loss = None
        hlens = torch.from_numpy(np.fromiter(hlens, dtype=np.int32))
        olens = torch.from_numpy(np.fromiter(
            (x.size(0) for x in ys), dtype=np.int32))

        # zero padding for hs
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))

        # zero padding for ys
        ys_true = torch.cat(ys).cpu().int()  # batch x olen

        # get length info
        logging.info(self.__class__.__name__ + ' input lengths:  ' + ''.join(str(hlens).split('\n')))
        logging.info(self.__class__.__name__ + ' output lengths: ' + ''.join(str(olens).split('\n')))

        # get ctc loss
        # expected shape of seqLength x batchSize x alphabet_size
        ys_hat = ys_hat.transpose(0, 1)
        self.loss = to_cuda(self, self.loss_fn(ys_hat, ys_true, hlens, olens))
        logging.info('ctc loss:' + str(float(self.loss)))

        return self.loss

    def log_softmax(self, hs_pad):
        '''log_softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        '''
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)


# ------------- Attention Network --------------------------------------------------------------------------------------
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

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B, T_max, D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights
        :rtype: torch.Tensor
        '''
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev = pad_list(att_prev, 0)
            self.c = torch.sum(self.enc_h * att_prev.view(batch, self.h_length, 1), dim=1)

        return self.c, att_prev


class AttDot(torch.nn.Module):
    '''Dot product attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    '''

    def __init__(self, eprojs, dunits, att_dim):
        super(AttDot, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weight (B x T_max)
        :rtype: torch.Tensor
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = torch.tanh(self.mlp_enc(self.enc_h))

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        e = torch.sum(self.pre_compute_enc_h * torch.tanh(self.mlp_dec(dec_z)).view(batch, 1, self.att_dim),
                      dim=2)  # utt x frame

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
        return c, w


class AttAdd(torch.nn.Module):
    '''Additive attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    '''

    def __init__(self, eprojs, dunits, att_dim):
        super(AttAdd, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights (B x T_max)
        :rtype: torch.Tensor
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, w


class AttLoc(torch.nn.Module):
    '''location-aware attention

    Reference: Attention-Based Models for Speech Recognition
        (https://arxiv.org/pdf/1506.07503.pdf)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.aconv_chans = aconv_chans

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attetion weight (B x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights (B x T_max)
        :rtype: torch.Tensor
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev = pad_list(att_prev, 0)

        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, w


class AttCov(torch.nn.Module):
    '''Coverage mechanism attention

    Reference: Get To The Point: Summarization with Pointer-Generator Network
       (https://arxiv.org/abs/1704.04368)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    '''

    def __init__(self, eprojs, dunits, att_dim):
        super(AttCov, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.wvec = torch.nn.Linear(1, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_list, scaling=2.0):
        '''AttCov forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param list att_prev_list: list of previous attetion weight
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weights
        :rtype: list
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev_list is None:
            att_prev = [enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev_list = [pad_list(att_prev, 0)]

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = sum(att_prev_list)
        # cov_vec: B x T => B x T x 1 => B x T x att_dim
        cov_vec = self.wvec(cov_vec.unsqueeze(-1))

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(cov_vec + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)
        att_prev_list += [w]

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, att_prev_list


class AttLoc2D(torch.nn.Module):
    '''2D location-aware attention

    This attention is an extended version of location aware attention.
    It take not only one frame before attention weights, but also earlier frames into account.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param int att_win: attention window size (default=5)
    '''

    def __init__(self, eprojs, dunits, att_dim, att_win, aconv_chans, aconv_filts):
        super(AttLoc2D, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (att_win, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans
        self.att_win = att_win
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttLoc2D forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attetion weight (B x att_win x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights (B x att_win x T_max)
        :rtype: torch.Tensor
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # B * [Li x att_win]
            att_prev = [enc_hs_pad.new(l, self.att_win).fill_(1.0 / l) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev = pad_list(att_prev, 0).transpose(1, 2)

        # att_prev: B x att_win x Tmax -> B x 1 x att_win x Tmax -> B x C x 1 x Tmax
        att_conv = self.loc_conv(att_prev.unsqueeze(1))
        # att_conv: B x C x 1 x Tmax -> B x Tmax x C
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        # update att_prev: B x att_win x Tmax -> B x att_win+1 x Tmax -> B x att_win x Tmax
        att_prev = torch.cat([att_prev, w.unsqueeze(1)], dim=1)
        att_prev = att_prev[:, 1:]

        return c, att_prev


class AttLocRec(torch.nn.Module):
    '''location-aware recurrent attention

    This attention is an extended version of location aware attention.
    With the use of RNN, it take the effect of the history of attention weights into account.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttLocRec, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.att_lstm = torch.nn.LSTMCell(aconv_chans, att_dim, bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_states, scaling=2.0):
        '''AttLocRec forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param tuple att_prev_states: previous attetion weight and lstm states
                                      ((B, T_max), ((B, att_dim), (B, att_dim)))
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights and lstm states (w, (hx, cx))
                 ((B, T_max), ((B, att_dim), (B, att_dim)))
        :rtype: tuple
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev_states is None:
            # initialize attention weight with uniform dist.
            att_prev = [enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev = pad_list(att_prev, 0)

            # initialize lstm states
            att_h = enc_hs_pad.new_zeros(batch, self.att_dim)
            att_c = enc_hs_pad.new_zeros(batch, self.att_dim)
            att_states = (att_h, att_c)
        else:
            att_prev = att_prev_states[0]
            att_states = att_prev_states[1]

        # B x 1 x 1 x T -> B x C x 1 x T
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # apply non-linear
        att_conv = F.relu(att_conv)
        # B x C x 1 x T -> B x C x 1 x 1 -> B x C
        att_conv = F.max_pool2d(att_conv, (1, att_conv.size(3))).view(batch, -1)

        att_h, att_c = self.att_lstm(att_conv, att_states)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_h.unsqueeze(1) + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, (w, (att_h, att_c))


class AttCovLoc(torch.nn.Module):
    '''Coverage mechanism location aware attention

    This attention is a combination of coverage and location-aware attentions.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttCovLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_list, scaling=2.0):
        '''AttCovLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param list att_prev_list: list of previous attetion weight
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weights
        :rtype: list
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev_list is None:
            att_prev = [enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev_list = [pad_list(att_prev, 0)]

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = sum(att_prev_list)

        # cov_vec: B x T -> B x 1 x 1 x T -> B x C x 1 x T
        att_conv = self.loc_conv(cov_vec.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)
        att_prev_list += [w]

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, att_prev_list


class AttMultiHeadDot(torch.nn.Module):
    '''Multi head dot product attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int ahead: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    '''

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v):
        super(AttMultiHeadDot, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''AttMultiHeadDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weight (B x T_max) * aheads
        :rtype: list
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                torch.tanh(self.mlp_k[h](self.enc_h)) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            e = torch.sum(self.pre_compute_k[h] * torch.tanh(self.mlp_q[h](dec_z)).view(
                batch, 1, self.att_dim_k), dim=2)  # utt x frame

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadAdd(torch.nn.Module):
    '''Multi head additive attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using additive attention for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int ahead: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    '''

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v):
        super(AttMultiHeadAdd, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''AttMultiHeadAdd forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weight (B x T_max) * aheads
        :rtype: list
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            e = self.gvec[h](torch.tanh(
                self.pre_compute_k[h] + self.mlp_q[h](dec_z).view(batch, 1, self.att_dim_k))).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadLoc(torch.nn.Module):
    '''Multi head location based attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using location-aware attention for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v, aconv_chans, aconv_filts):
        super(AttMultiHeadLoc, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        self.loc_conv = torch.nn.ModuleList()
        self.mlp_att = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
            self.loc_conv += [torch.nn.Conv2d(
                1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)]
            self.mlp_att += [torch.nn.Linear(aconv_chans, att_dim_k, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttMultiHeadLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: list of previous attentioin weight (B x T_max) * aheads
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weight (B x T_max) * aheads
        :rtype: list
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            att_prev = []
            for h in six.moves.range(self.aheads):
                att_prev += [[enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]]
                # if no bias, 0 0-pad goes 0
                att_prev[h] = pad_list(att_prev[h], 0)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch, 1, 1, self.h_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = self.mlp_att[h](att_conv)

            e = self.gvec[h](torch.tanh(
                self.pre_compute_k[h] + att_conv + self.mlp_q[h](dec_z).view(
                    batch, 1, self.att_dim_k))).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadMultiResLoc(torch.nn.Module):
    '''Multi head multi resolution location based attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using location-aware attention for each head.
    Furthermore, it uses different filter size for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    :param int aconv_chans: maximum # channels of attention convolution
        each head use #ch = aconv_chans * (head + 1) / aheads
        e.g. aheads=4, aconv_chans=100 => filter size = 25, 50, 75, 100
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v, aconv_chans, aconv_filts):
        super(AttMultiHeadMultiResLoc, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        self.loc_conv = torch.nn.ModuleList()
        self.mlp_att = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
            afilts = aconv_filts * (h + 1) // aheads
            self.loc_conv += [torch.nn.Conv2d(
                1, aconv_chans, (1, 2 * afilts + 1), padding=(0, afilts), bias=False)]
            self.mlp_att += [torch.nn.Linear(aconv_chans, att_dim_k, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''AttMultiHeadMultiResLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: list of previous attentioin weight (B x T_max) * aheads
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weight (B x T_max) * aheads
        :rtype: list
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            att_prev = []
            for h in six.moves.range(self.aheads):
                att_prev += [[enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]]
                # if no bias, 0 0-pad goes 0
                att_prev[h] = pad_list(att_prev[h], 0)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch, 1, 1, self.h_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = self.mlp_att[h](att_conv)

            e = self.gvec[h](torch.tanh(
                self.pre_compute_k[h] + att_conv + self.mlp_q[h](dec_z).view(
                    batch, 1, self.att_dim_k))).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttForward(torch.nn.Module):
    '''Forward attention

    Reference: Forward attention in sequence-to-sequence acoustic modeling for speech synthesis
        (https://arxiv.org/pdf/1807.06736.pdf)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    '''

    def __init__(self, eprojs, dunits, att_dim):
        super(AttForward, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=1.0):
        '''AttForward forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: attention weights of previous step
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights (B x T_max)
        :rtype: torch.Tensor
        '''
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            # initial attention will be [1, 0, 0, ...]
            att_prev = enc_hs_pad.new_zeros(*enc_hs_pad.size()[:2])
            att_prev[:, 0] = 1.0

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).unsqueeze(1)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # forward attention
        att_prev_shift = F.pad(att_prev, (1, 0))[:, :-1]
        w = (att_prev + att_prev_shift) * w
        # NOTE: clamp is needed to avoid nan gradient
        w = F.normalize(torch.clamp(w, 1e-6), p=1, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.unsqueeze(-1), dim=1)

        return c, w


class AttForwardTA(torch.nn.Module):
    '''Forward attention with transition agent

    Reference: Forward attention in sequence-to-sequence acoustic modeling for speech synthesis
        (https://arxiv.org/pdf/1807.06736.pdf)

    :param int eunits: # units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int odim: output dimension
    '''

    def __init__(self, eunits, dunits, att_dim, odim):
        super(AttForwardTA, self).__init__()
        self.mlp_enc = torch.nn.Linear(eunits, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_ta = torch.nn.Linear(eunits + dunits + odim, 1)
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eunits = eunits
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.trans_agent_prob = 0.5

    def reset(self):
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.trans_agent_prob = 0.5

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, out_prev, scaling=1.0):
        '''AttForwardTA forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B, Tmax, eunits)
        :param list enc_hs_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B, dunits)
        :param torch.Tensor att_prev: attention weights of previous step
        :param torch.Tensor prev_out: decoder outputs of previous step (B, odim)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, dunits)
        :rtype: torch.Tensor
        :return: previous attentioin weights (B, Tmax)
        :rtype: torch.Tensor
        '''
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            # initial attention will be [1, 0, 0, ...]
            att_prev = enc_hs_pad.new_zeros(*enc_hs_pad.size()[:2])
            att_prev[:, 0] = 1.0

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # forward attention
        att_prev_shift = F.pad(att_prev, (1, 0))[:, :-1]
        w = (self.trans_agent_prob * att_prev + (1 - self.trans_agent_prob) * att_prev_shift) * w
        # NOTE: clamp is needed to avoid nan gradient
        w = F.normalize(torch.clamp(w, 1e-6), p=1, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        # update transition agent prob
        self.trans_agent_prob = torch.sigmoid(
            self.mlp_ta(torch.cat([c, out_prev, dec_z], dim=1)))

        return c, w


# ------------- Decoder Network ----------------------------------------------------------------------------------------
class Decoder(torch.nn.Module):
    """Decoder module

    :param int eprojs: # encoder projection units
    :param int odim: dimension of outputs
    :param int dlayers: # decoder layers
    :param int dunits: # decoder units
    :param int sos: start of sequence symbol id
    :param int eos: end of sequence symbol id
    :param torch.nn.Module att: attention module
    :param int verbose: verbose level
    :param list char_list: list of character strings
    :param ndarray labeldist: distribution of label smoothing
    :param float lsm_weight: label smoothing weight
    """

    def __init__(self, eprojs, odim, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0.):
        super(Decoder, self).__init__()
        self.dunits = dunits
        self.dlayers = dlayers
        self.embed = torch.nn.Embedding(odim, dunits)
        self.decoder = torch.nn.ModuleList()
        self.decoder += [torch.nn.LSTMCell(dunits + eprojs, dunits)]
        for l in six.moves.range(1, self.dlayers):
            self.decoder += [torch.nn.LSTMCell(dunits, dunits)]
        self.ignore_id = -1
        self.output = torch.nn.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def forward(self, hs_pad, hlens, ys_pad):
        '''Decoder forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        '''
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

        # hlen should be list of integer
        hlens = list(map(int, hlens))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get dim, length info
        batch = ys_out_pad.size(0)
        olength = ys_out_pad.size(1)
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(hlens))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out]))

        # initialization
        c_list = [self.zero_state(hs_pad)]
        z_list = [self.zero_state(hs_pad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad))
            z_list.append(self.zero_state(hs_pad))
        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(ys_in_pad)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs_pad, hlens, z_list[0], att_w)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            z_all.append(z_list[-1])

        z_all = torch.stack(z_all, dim=1).view(batch * olength, self.dunits)
        # compute loss
        y_all = self.output(z_all)
        self.loss = F.cross_entropy(y_all, ys_out_pad.view(-1),
                                    ignore_index=self.ignore_id,
                                    size_average=True)
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = th_accuracy(y_all, ys_out_pad, ignore_label=self.ignore_id)
        logging.info('att loss:' + ''.join(str(self.loss.item()).split('\n')))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            ys_hat = y_all.view(batch, olength, -1)
            ys_true = ys_out_pad
            for (i, y_hat), y_true in zip(enumerate(ys_hat.detach().cpu().numpy()),
                                          ys_true.detach().cpu().numpy()):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat[y_true != self.ignore_id], axis=1)
                idx_true = y_true[y_true != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_cuda(self, torch.from_numpy(self.labeldist))
            loss_reg = - torch.sum((F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None):
        '''beam search implementation

        :param torch.Tensor h: encoder hidden state (T, eprojs)
        :param torch.Tensor lpz: ctc log softmax output (T, odim)
        :param Namespace recog_args: argument namespace contraining options
        :param char_list: list of character strings
        :param torch.nn.Module rnnlm: language module
        :return: N-best decoding results
        :rtype: list of dicts
        '''
        logging.info('input lengths: ' + str(h.size(0)))
        # initialization
        c_list = [self.zero_state(h.unsqueeze(0))]
        z_list = [self.zero_state(h.unsqueeze(0))]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(h.unsqueeze(0)))
            z_list.append(self.zero_state(h.unsqueeze(0)))
        a = None
        self.att.reset()  # reset pre-computation of h

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprate sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list,
                   'z_prev': z_list, 'a_prev': a, 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, np)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]
                ey = self.embed(vy)           # utt list (1) x zdim
                ey.unsqueeze(0)
                att_c, att_w = self.att(h.unsqueeze(0), [h.size(0)], hyp['z_prev'][0], hyp['a_prev'])
                ey = torch.cat((ey, att_c), dim=1)   # utt(1) x (zdim + hdim)
                z_list[0], c_list[0] = self.decoder[0](ey, (hyp['z_prev'][0], hyp['c_prev'][0]))
                for l in six.moves.range(1, self.dlayers):
                    z_list[l], c_list[l] = self.decoder[l](
                        z_list[l - 1], (hyp['z_prev'][l], hyp['c_prev'][l]))

                # get nbest local scores and their ids
                local_att_scores = F.log_softmax(self.output(z_list[-1]), dim=1)
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # [:] is needed!
                    new_hyp['z_prev'] = z_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_w[:]
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            logging.debug(
                'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug(
                    'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warn('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize_beam(h, lpz, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        # remove sos
        return nbest_hyps

    def calculate_all_attentions(self, hs_pad, hlen, ys_pad):
        '''Calculate all of attentions

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        '''
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

        # hlen should be list of integer
        hlen = list(map(int, hlen))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get length info
        olength = ys_out_pad.size(1)

        # initialization
        c_list = [self.zero_state(hs_pad)]
        z_list = [self.zero_state(hs_pad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad))
            z_list.append(self.zero_state(hs_pad))
        att_w = None
        att_ws = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(ys_in_pad)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs_pad, hlen, z_list[0], att_w)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            att_ws.append(att_w)

        # convert to numpy array with the shape (B, Lmax, Tmax)
        if isinstance(self.att, AttLoc2D):
            # att_ws => list of previous concate attentions
            att_ws = torch.stack([aw[:, -1] for aw in att_ws], dim=1).cpu().numpy()
        elif isinstance(self.att, (AttCov, AttCovLoc)):
            # att_ws => list of list of previous attentions
            att_ws = torch.stack([aw[-1] for aw in att_ws], dim=1).cpu().numpy()
        elif isinstance(self.att, AttLocRec):
            # att_ws => list of tuple of attention and hidden states
            att_ws = torch.stack([aw[0] for aw in att_ws], dim=1).cpu().numpy()
        elif isinstance(self.att, (AttMultiHeadDot, AttMultiHeadAdd, AttMultiHeadLoc, AttMultiHeadMultiResLoc)):
            # att_ws => list of list of each head attetion
            n_heads = len(att_ws[0])
            att_ws_sorted_by_head = []
            for h in six.moves.range(n_heads):
                att_ws_head = torch.stack([aw[h] for aw in att_ws], dim=1)
                att_ws_sorted_by_head += [att_ws_head]
            att_ws = torch.stack(att_ws_sorted_by_head, dim=1).cpu().numpy()
        else:
            # att_ws => list of attetions
            att_ws = torch.stack(att_ws, dim=1).cpu().numpy()
        return att_ws


# ------------- Encoder Network ----------------------------------------------------------------------------------------
class Encoder(torch.nn.Module):
    '''Encoder module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int epojs: number of projection units of encoder network
    :param list subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
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
            self.enc2 = BLSTMP(get_vgg2l_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            logging.info('Use CNN-VGG + BLSTMP for encoder')
        elif etype == 'vggblstm':
            self.enc1 = VGG2L(in_channel)
            self.enc2 = BLSTM(get_vgg2l_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG + BLSTM for encoder')
        else:
            logging.error(
                "Error: need to specify an appropriate encoder archtecture")
            sys.exit()

        self.etype = etype

    def forward(self, xs_pad, ilens):
        '''Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, erojs)
        :rtype: torch.Tensor
        '''
        if self.etype == 'blstm':
            xs_pad, ilens = self.enc1(xs_pad, ilens)
        elif self.etype == 'blstmp':
            xs_pad, ilens = self.enc1(xs_pad, ilens)
        elif self.etype == 'vggblstmp':
            xs_pad, ilens = self.enc1(xs_pad, ilens)
            xs_pad, ilens = self.enc2(xs_pad, ilens)
        elif self.etype == 'vggblstm':
            xs_pad, ilens = self.enc1(xs_pad, ilens)
            xs_pad, ilens = self.enc2(xs_pad, ilens)
        else:
            logging.error(
                "Error: need to specify an appropriate encoder archtecture")
            sys.exit()

        # make mask to remove bias value in padded part
        mask = to_cuda(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad.masked_fill(mask, 0.0), ilens


class BLSTMP(torch.nn.Module):
    """Bidirectional LSTM with projection layer module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of lstm units (resulted in cdim * 2 due to biderectional)
    :param int hdim: number of projection units
    :param list subsample: list of subsampling numbers
    :param float dropout: dropout rate
    """

    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout):
        super(BLSTMP, self).__init__()
        for i in six.moves.range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim
            setattr(self, "bilstm%d" % i, torch.nn.LSTM(inputdim, cdim, dropout=dropout,
                                                        num_layers=1, bidirectional=True, batch_first=True))
            # bottleneck layer to merge
            setattr(self, "bt%d" % i, torch.nn.Linear(2 * cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample

    def forward(self, xs_pad, ilens):
        '''BLSTMP forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, hdim)
        :rtype: torch.Tensor
        '''
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        for layer in six.moves.range(self.elayers):
            xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
            bilstm = getattr(self, 'bilstm' + str(layer))
            bilstm.flatten_parameters()
            ys, _ = bilstm(xs_pack)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ys_pad = ys_pad[:, ::sub]
                ilens = [int(i + 1) // sub for i in ilens]
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'bt' + str(layer)
                                )(ys_pad.contiguous().view(-1, ys_pad.size(2)))
            xs_pad = torch.tanh(projected.view(ys_pad.size(0), ys_pad.size(1), -1))

        return xs_pad, ilens  # x: utt list of frame x dim


class BLSTM(torch.nn.Module):
    """Bidirectional LSTM module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of lstm units (resulted in cdim * 2 due to biderectional)
    :param int hdim: number of final projection units
    :param list subsample: list of subsampling numbers
    :param float dropout: dropout rate
    """

    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(BLSTM, self).__init__()
        self.nblstm = torch.nn.LSTM(idim, cdim, elayers, batch_first=True,
                                    dropout=dropout, bidirectional=True)
        self.l_last = torch.nn.Linear(cdim * 2, hdim)

    def forward(self, xs_pad, ilens):
        '''BLSTM forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, erojs)
        :rtype: torch.Tensor
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        self.nblstm.flatten_parameters()
        ys, _ = self.nblstm(xs_pack)
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
        # (sum _utt frame_utt) x dim
        projected = torch.tanh(self.l_last(
            ys_pad.contiguous().view(-1, ys_pad.size(2))))
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
        '''VGG2L forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of padded hidden state sequences (B, Tmax // 4, 128)
        :rtype: torch.Tensor
        '''
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
        xs_pad = [xs_pad[i, :ilens[i]] for i in range(len(ilens))]
        xs_pad = pad_list(xs_pad, 0.0)
        return xs_pad, ilens
