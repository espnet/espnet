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
        self.ctc = CTC(odim, args.eprojs, args.dropout_rate)
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
        :return: ctc loass value
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
        self.loss_fn = warp_ctc.CTCLoss(size_average=True, reduce=False)
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

        return self.loss

    def log_softmax(self, hs_pad):
        '''log_softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        '''
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)


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
                 char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0):
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
        self.sampling_probability = sampling_probability

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def forward(self, hs_pad, hlens, ys_pad, strm_idx):
        '''Decoder forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :param int strm_idx: stream index for attention list when args.spa (speaker parallel attention) is True
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        '''
        # TODO(kan-bayashi): need to make more smart way
        att_idx = min(strm_idx, len(self.att)-1)
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
        self.att[att_idx].reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(ys_in_pad)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att[att_idx](hs_pad, hlens, z_list[0], att_w)
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(' scheduled sampling ')
                z_out = self.output(z_all[-1])
                z_out = np.argmax(z_out.detach(), axis=1)
                z_out = self.embed(z_out.cuda())
                ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
            else:
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

    def recognize_beam(self, h, lpz, recog_args, char_list, strm_idx, rnnlm=None):
        '''beam search implementation

        :param torch.Tensor h: encoder hidden state (T, eprojs)
        :param torch.Tensor lpz: ctc log softmax output (T, odim)
        :param Namespace recog_args: argument namespace contraining options
        :param char_list: list of character strings
        :param int strm_idx: stream index for speaker parallel attention
        :param torch.nn.Module rnnlm: language module
        :return: N-best decoding results
        :rtype: list of dicts
        '''
        logging.info('input lengths: ' + str(h.size(0)))
        att_idx = min(strm_idx, len(self.att)-1)
        # initialization
        c_list = [self.zero_state(h.unsqueeze(0))]
        z_list = [self.zero_state(h.unsqueeze(0))]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(h.unsqueeze(0)))
            z_list.append(self.zero_state(h.unsqueeze(0)))
        a = None
        self.att[att_idx].reset()  # reset pre-computation of h

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
                att_c, att_w = self.att[att_idx](h.unsqueeze(0), [h.size(0)], hyp['z_prev'][0], hyp['a_prev'])
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
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
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

    def calculate_all_attentions(self, hs_pad, hlen, ys_pad, strm_idx):
        '''Calculate all of attentions

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :param int strm_idx: stream index for speaker parallel attention
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        '''
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        att_idx = min(strm_idx, len(self.att)-1)

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
        self.att[att_idx].reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(ys_in_pad)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att[att_idx](hs_pad, hlen, z_list[0], att_w)
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
