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
import random
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
from e2e_asr_common import mix_label_smoothing_dist

# reuse e2e_asr_th
from e2e_asr_th import to_cuda
from e2e_asr_th import pad_list
from e2e_asr_th import make_pad_mask
from e2e_asr_th import th_accuracy
from e2e_asr_th import Reporter
from e2e_asr_th import Loss
from e2e_asr_th import CTC
from e2e_asr_th import Decoder
from e2e_asr_th import NoAtt, AttDot, AttAdd, AttLoc, AttCov, AttLoc2D, AttLocRec, AttCovLoc, AttMultiHeadDot, AttMultiHeadAdd, AttMultiHeadLoc, AttMultiHeadMultiResLoc, AttForward, AttForwardTA
from e2e_asr_th import BLSTMP, BLSTM, VGG2L

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5


# ------------- Utility functions --------------------------------------------------------------------------------------
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
        self.num_spkrs = args.num_spkrs
        self.spa = args.spa

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers_sd + args.elayers_rec + 1, dtype=np.int)
        if args.etype == 'blstmp':
            ss = args.subsample.split("_")
            for j in range(min(args.elayers_sd + args.elayers_rec + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args.lsm_type:
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = mix_label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # encoder
        self.enc = Encoder(args.etype, idim, args.elayers_sd, args.elayers_rec, args.eunits,
                           args.eprojs, self.subsample, args.dropout_rate, num_spkrs=args.num_spkrs)
        # ctc
        self.ctc = CTC(odim, args.eprojs, args.dropout_rate, reduce=False)
        # attention
        self.att = torch.nn.ModuleList()
        num_att = self.num_spkrs if args.spa else 1
        for i in range(num_att):
            if args.atype == 'noatt':
                att = NoAtt()
            elif args.atype == 'dot':
                att = AttDot(args.eprojs, args.dunits, args.adim)
            elif args.atype == 'add':
                att = AttAdd(args.eprojs, args.dunits, args.adim)
            elif args.atype == 'location':
                att = AttLoc(args.eprojs, args.dunits,
                             args.adim, args.aconv_chans, args.aconv_filts)
            elif args.atype == 'location2d':
                att = AttLoc2D(args.eprojs, args.dunits,
                               args.adim, args.awin, args.aconv_chans, args.aconv_filts)
            elif args.atype == 'location_recurrent':
                att = AttLocRec(args.eprojs, args.dunits,
                                args.adim, args.aconv_chans, args.aconv_filts)
            elif args.atype == 'coverage':
                att = AttCov(args.eprojs, args.dunits, args.adim)
            elif args.atype == 'coverage_location':
                att = AttCovLoc(args.eprojs, args.dunits, args.adim,
                                args.aconv_chans, args.aconv_filts)
            elif args.atype == 'multi_head_dot':
                att = AttMultiHeadDot(args.eprojs, args.dunits,
                                      args.aheads, args.adim, args.adim)
            elif args.atype == 'multi_head_add':
                att = AttMultiHeadAdd(args.eprojs, args.dunits,
                                      args.aheads, args.adim, args.adim)
            elif args.atype == 'multi_head_loc':
                att = AttMultiHeadLoc(args.eprojs, args.dunits,
                                      args.aheads, args.adim, args.adim,
                                      args.aconv_chans, args.aconv_filts)
            elif args.atype == 'multi_head_multi_res_loc':
                att = AttMultiHeadMultiResLoc(args.eprojs, args.dunits,
                                              args.aheads, args.adim, args.adim,
                                              args.aconv_chans, args.aconv_filts)
            else:
                logging.error(
                    "Error: need to specify an appropriate attention archtecture")
                sys.exit()
            self.att.append(att)
        # decoder
        self.dec = Decoder(args.eprojs, odim, args.dlayers, args.dunits,
                           self.sos, self.eos, self.att, self.verbose, self.char_list,
                           labeldist, args.lsm_weight, args.sampling_probability)

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

    def min_PIT_score(self, losses):
        '''E2E min_PIT_score

        :param torch.Tensor losses: list of losses
        :return: min ctc loss value
        :rtype: torch.Tensor
        :return: permutation of min ctc loss value
        :rtype: torch.Tensor
        '''
        if self.num_spkrs == 1:
            return losses[0], torch.zeros(losses[0].size(0), dtype=torch.long)
        elif self.num_spkrs == 2:
            batch_sz = losses[0].size(0)
            loss_perm = to_cuda(self, torch.zeros(batch_sz, dtype=losses[0][0].dtype))
            permutation = torch.zeros(batch_sz, 2, dtype=torch.long, requires_grad=False)
            for i in range(batch_sz):
                score_perm1 = (losses[0][i] + losses[3][i]) / 2
                score_perm2 = (losses[1][i] + losses[2][i]) / 2
                if (score_perm1 <= score_perm2):
                    loss_perm[i] = score_perm1
                    permutation[i] = torch.LongTensor([0,1])
                    losses[1][i].detach(), losses[2][i].detach()
                else:
                    loss_perm[i] = score_perm2
                    permutation[i] = torch.LongTensor([1,0])
                    losses[0][i].detach(), losses[3][i].detach()
            return torch.mean(loss_perm), permutation
        elif self.num_spkrs == 3:
            batch_sz = losses[0].size(0)
            loss_perm = to_cuda(self, torch.zeros(batch_sz, dtype=losses[0][0].dtype))
            perm_choices = [[0,1,2],[0,2,1],[1,2,0],[1,0,2],[2,0,1],[2,1,0]]
            permutation = torch.zeros(batch_sz, 6, dtype=torch.long, requires_grad=False)
            for i in range(batch_sz):
                loss_perm[i], min_idx = torch.tensor([(losses[0][i] + losses[4][i] + losses[8][i]) / 3,
                                 (losses[0][i] + losses[5][i] + losses[7][i]) / 3,
                                 (losses[1][i] + losses[5][i] + losses[6][i]) / 3,
                                 (losses[1][i] + losses[3][i] + losses[8][i]) / 3,
                                 (losses[2][i] + losses[3][i] + losses[7][i]) / 3,
                                 (losses[2][i] + losses[4][i] + losses[6][i]) / 3], dtype=losses[0].dtype, device=losses[0].device, requires_grad=losses[0].requires_grad).min()
                permutation[i] = perm_choices[min_idx]
                # there may be some problem in detach these variables
                for j in range(8):
                    losses[j][i].detach()
            return torch.mean(loss_perm), permutation
        else:
            logging.error("Error: support no more than 2 speakers.")
            sys.exit()


    def forward(self, xs_pad, ilens, ys_pad_sd):
        '''E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad_sd: batch of padded character id sequence tensor (B, num_spkrs, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        '''
        # 1. encoder
        hs_pad_sd, hlens = self.enc(xs_pad, ilens)

        # 2. CTC loss
        ys_pad_sd = ys_pad_sd.transpose(0, 1) # (num_spkrs, B, Lmax)
        if self.mtlalpha == 0:
            loss_ctc, min_perm = None, None
            logging.error("Error: CTC has to be computed for permutation.")
            sys.exit()
        elif self.num_spkrs <= 3:
            loss_ctc = [self.ctc(hs_pad_sd[i // self.num_spkrs], hlens,
                                 ys_pad_sd[i % self.num_spkrs]) for i in range(self.num_spkrs ** 2)]
            loss_ctc, min_perm = self.min_PIT_score(loss_ctc)
            logging.info('ctc loss:' + str(float(loss_ctc)))

        # 3. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
        else:
            for i in range(ys_pad_sd.size(1)):
                ys_pad_sd[:,i] = ys_pad_sd[min_perm[i], i]
            rslt = [self.dec(hs_pad_sd[i], hlens, ys_pad_sd[i], i) for i in range(self.num_spkrs)]
            loss_att = sum([r[0] for r in rslt]) / float(len(rslt))
            acc      = sum([r[1] for r in rslt]) / float(len(rslt))

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
        h = h.contiguous()
        h_sd, _ = self.enc(h.unsqueeze(0), ilen)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = [self.ctc.log_softmax(h)[0] for h in h_sd]
        else:
            lpz = None

        # 2. decoder
        # decode the first utterance
        y = [self.dec.recognize_beam(h_sd[i][0], lpz[i], recog_args, char_list, i, rnnlm) for i in range(self.num_spkrs)]

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad_sd):
        '''E2E attention calculation

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad_sd: batch of padded character id sequence tensor (B, num_spkrs, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        '''
        with torch.no_grad():
            # encoder
            hpad_sd, hlens = self.enc(xs_pad, ilens)

            ys_pad_sd = ys_pad_sd.transpose(0, 1) # (num_spkrs, B, Lmax)
            # decoder
            att_ws_sd = [self.dec.calculate_all_attentions(hpad_sd[i], hlens, ys_pad_sd[i], i) for i in range(self.num_spkrs)]

        return att_ws_sd


# ------------- Encoder Network ----------------------------------------------------------------------------------------
class Encoder(torch.nn.Module):
    '''Encoder module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers_sd: number of layers of speaker differentiate part in encoder network
    :param int elayers_rec: number of layers of shared recognition part in encoder network
    :param int eunits: number of lstm units of encoder network
    :param int epojs: number of projection units of encoder network
    :param list subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    :param int num_spkrs: number of speakers
    '''

    def __init__(self, etype, idim, elayers_sd, elayers_rec, eunits, eprojs, subsample, dropout, in_channel=1, num_spkrs=2):
        super(Encoder, self).__init__()

        if etype == 'vggblstmp':
            self.enc1 = VGG2L(in_channel)
            self.enc_sd = torch.nn.ModuleList([BLSTMP(get_vgg2l_odim(idim, in_channel=in_channel),
                               elayers_sd, eunits, eprojs,
                               subsample[:elayers_sd+1], dropout) for i in range(num_spkrs)]) # leave subsample[0] for the input due to the design of BLSTMP
            self.enc_rec = BLSTMP(eprojs, elayers_rec, eunits, eprojs, subsample[elayers_sd:], dropout)
            logging.info('Use CNN-VGG + BLSTMP for encoder')
        else:
            logging.error(
                "Error: need to specify an appropriate encoder archtecture")
            sys.exit()

        self.etype = etype
        self.num_spkrs = num_spkrs

    def forward(self, xs_pad, ilens):
        '''Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, erojs)
        :rtype: torch.Tensor
        '''
        xs_pad_sd = [None for i in range(self.num_spkrs)]
        ilens_sd  = [None for i in range(self.num_spkrs)]
        if self.etype == 'vggblstmp':
            xs_pad, ilens = self.enc1(xs_pad, ilens)
            for i in range(self.num_spkrs):
                xs_pad_sd[i], ilens = self.enc_rec(*self.enc_sd[i](xs_pad, ilens))
        else:
            logging.error(
                "Error: need to specify an appropriate encoder archtecture")
            sys.exit()

        # make mask to remove bias value in padded part
        mask = to_cuda(self, make_pad_mask(ilens).unsqueeze(-1))

        return [x.masked_fill(mask, 0.0) for x in xs_pad_sd], ilens
