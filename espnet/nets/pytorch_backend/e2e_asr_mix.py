#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import argparse
import logging
import math
import sys

import editdistance

import chainer
import numpy as np
import six
import torch

from chainer import reporter

from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.e2e_asr_common import label_smoothing_dist

from espnet.nets.pytorch_backend.attentions import att_for
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.decoders import decoder_for
from espnet.nets.pytorch_backend.encoders import RNNP
from espnet.nets.pytorch_backend.encoders import VGG2L

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device

CTC_LOSS_THRESHOLD = 10000


class Reporter(chainer.Chain):
    """A chainer reporter wrapper"""

    def report(self, loss_ctc, loss_att, acc, cer, wer, mtl_loss):
        reporter.report({'loss_ctc': loss_ctc}, self)
        reporter.report({'loss_att': loss_att}, self)
        reporter.report({'acc': acc}, self)
        reporter.report({'cer': cer}, self)
        reporter.report({'wer': wer}, self)
        logging.info('mtl loss:' + str(mtl_loss))
        reporter.report({'loss': mtl_loss}, self)


class E2E(torch.nn.Module):
    """E2E module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    """

    def __init__(self, idim, odim, args):
        super(E2E, self).__init__()
        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.reporter = Reporter()
        self.num_spkrs = args.num_spkrs
        self.spa = args.spa

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer_sd + args.elayers)
        subsample = np.ones(args.elayers_sd + args.elayers + 1, dtype=np.int)
        if args.etype.endswith("p") and not args.etype.startswith("vgg"):
            ss = args.subsample.split("_")
            for j in range(min(args.elayers_sd + args.elayers + 1, len(ss))):
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
        self.enc = encoder_for(args, idim, self.subsample)
        # ctc
        self.ctc = ctc_for(args, odim, reduce=False)
        # attention
        num_att = self.num_spkrs if args.spa else 1
        self.att = att_for(args, num_att)
        # decoder
        self.dec = decoder_for(args, odim, self.sos, self.eos, self.att, labeldist)

        # weight initialization
        self.init_like_chainer()

        # options for beam search
        if 'report_cer' in vars(args) and (args.report_cer or args.report_wer):
            recog_args = {'beam_size': args.beam_size, 'penalty': args.penalty,
                          'ctc_weight': args.ctc_weight, 'maxlenratio': args.maxlenratio,
                          'minlenratio': args.minlenratio, 'lm_weight': args.lm_weight,
                          'rnnlm': args.rnnlm, 'nbest': args.nbest,
                          'space': args.sym_space, 'blank': args.sym_blank}

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.loss = None
        self.acc = None

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

    def min_pit_process(self, loss):
        '''E2E min_pit_process

        :param list|1-D torch.Tensor loss: list of losses for each sample [h1r1,h1r2,h2r1,h2r2],
            or [h1r1,h1r2,h1r3,h2r1,h2r2,h2r3,h3r1,h3r2,h3r3]
        :return: min_loss
        :rtype: torch.Tensor size=(2|3)
        :return: permutation
        :rtype: List, len=(2|3)
        '''
        if self.num_spkrs == 2:
            perm_choices = [[0, 1], [1, 0]]
            score_perms = torch.stack([loss[0] + loss[3],
                                       loss[1] + loss[2]]) / self.num_spkrs
        elif self.num_spkrs == 3:
            perm_choices = [[0, 1, 2], [0, 2, 1], [1, 2, 0], [1, 0, 2], [2, 0, 1], [2, 1, 0]]
            score_perms = torch.stack([loss[0] + loss[4] + loss[8],
                                       loss[0] + loss[5] + loss[7],
                                       loss[1] + loss[5] + loss[6],
                                       loss[1] + loss[3] + loss[8],
                                       loss[2] + loss[3] + loss[7],
                                       loss[2] + loss[4] + loss[6]]) / self.num_spkrs
        else:
            raise Exception("NotImplementedError")

        perm_loss, min_idx = torch.min(score_perms, 0)
        permutation = perm_choices[min_idx]

        return perm_loss, permutation

    def min_pit_ctc_batch(self, losses):
        '''E2E min_pit_ctc_batch

        :param torch.Tensor losses: CTC losses (B, 1|4|9)
        :return: min ctc loss value
        :rtype: torch.Tensor (B)
        :return: permutation of min ctc loss value
        :rtype: torch.Tensor (B, 1|2|3)
        '''
        if self.num_spkrs == 1:
            return to_device(self, torch.mean(losses[:, 0])), to_device(self, torch.zeros(losses.size(0)).long())
        else:
            bs = losses.size(0)
            ret = [self.min_pit_process(losses[i]) for i in range(bs)]
            loss_perm = torch.stack([r[0] for r in ret], dim=0)
            permutation = torch.tensor([r[1] for r in ret]).long()
            return torch.mean(to_device(self, loss_perm)), to_device(self, permutation)

    def forward(self, xs_pad, ilens, ys_pad_sd):
        """E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad_sd: batch of padded character id sequence tensor (B, num_spkrs, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. encoder
        hs_pad_sd, hlens = self.enc(xs_pad, ilens)

        # 2. CTC loss
        ys_pad_sd = ys_pad_sd.transpose(0, 1)  # (num_spkrs, B, Lmax)
        if self.mtlalpha == 0:
            loss_ctc, min_perm = None, None
        elif self.num_spkrs <= 3:
            loss_ctc_perm = torch.stack([self.ctc(hs_pad_sd[i // self.num_spkrs], hlens, ys_pad_sd[i % self.num_spkrs])
                                         for i in range(self.num_spkrs ** 2)], dim=1)  # (B, num_spkrs^2)
            loss_ctc, min_perm = self.min_pit_ctc_batch(loss_ctc_perm)
            logging.info('ctc loss:' + str(float(loss_ctc)))

        # 3. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
        else:
            for i in range(ys_pad_sd.size(1)):  # B
                ys_pad_sd[:, i] = ys_pad_sd[min_perm[i], i]
            rslt = [self.dec(hs_pad_sd[i], hlens, ys_pad_sd[i], strm_idx=i) for i in range(self.num_spkrs)]
            loss_att = sum([r[0] for r in rslt]) / float(len(rslt))
            acc = sum([r[1] for r in rslt]) / float(len(rslt))
        self.acc = acc

        # 5. compute cer/wer
        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz_sd = [self.ctc.log_softmax(hs_pad_sd[i]).data for i in range(self.num_spkrs)]
            else:
                lpz_sd = None

            wers, cers = [], []
            nbest_hyps_sd = [self.dec.recognize_beam_batch(hs_pad_sd[i], torch.tensor(hlens), lpz_sd[i],
                                                           self.recog_args, self.char_list, self.rnnlm, strm_idx=i)
                             for i in range(self.num_spkrs)]
            # remove <sos> and <eos>
            y_hats_sd = [[nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps_sd[i]] for i in range(self.num_spkrs)]
            for i in range(len(y_hats_sd[0])):
                hyp_words = []
                hyp_chars = []
                ref_words = []
                ref_chars = []
                for ns in range(self.num_spkrs):
                    y_hat = y_hats_sd[ns][i]
                    y_true = ys_pad_sd[ns][i]

                    seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                    seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                    seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                    seq_hat_text = seq_hat_text.replace(self.recog_args.blank, '')
                    seq_true_text = "".join(seq_true).replace(self.recog_args.space, ' ')

                    hyp_words.append(seq_hat_text.split())
                    ref_words.append(seq_true_text.split())
                    hyp_chars.append(seq_hat_text.replace(' ', ''))
                    ref_chars.append(seq_true_text.replace(' ', ''))

                tmp_wers = [editdistance.eval(hyp_words[ns // self.num_spkrs], ref_words[ns % self.num_spkrs])
                            for ns in range(self.num_spkrs)]  # h1r1,h1r2,h2r1,h2r2
                tmp_cers = [editdistance.eval(hyp_chars[ns // self.num_spkrs], ref_chars[ns % self.num_spkrs])
                            for ns in range(self.num_spkrs)]  # h1r1,h1r2,h2r1,h2r2

                wers.append(self.min_pit_process(tmp_wers) / len(sum(ref_words, [])))
                cers.append(self.min_pit_process(tmp_cers) / len(sum(ref_words, [])))

            wer = 0.0 if not self.report_wer else sum(wers) / len(wers)
            cer = 0.0 if not self.report_cer else sum(cers) / len(cers)

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
            self.reporter.report(loss_ctc_data, loss_att_data, acc, cer, wer, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        # Note(kamo): In order to work with DataParallel, on pytorch==0.4,
        # the return value must be torch.CudaTensor, or tuple/list/dict of it.
        # Neither CPUTensor nor float/int value can be used
        # because NCCL communicates between GPU devices.
        device = next(self.parameters()).device

        acc = torch.tensor([acc], device=device) if acc is not None else None
        cer = torch.tensor([cer], device=device)
        wer = torch.tensor([wer], device=device)
        return self.loss, loss_ctc, loss_att, acc, cer, wer

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        """E2E beam search

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))

        # 1. encoder
        # make a utt list (1) to use the same interface for encoder
        h = h.contiguous()
        h_sd, _ = self.enc(h.unsqueeze(0), ilen)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz_sd = [self.ctc.log_softmax(i)[0] for i in h_sd]
        else:
            lpz_sd = None

        # 2. decoder
        # decode the first utterance
        y = [self.dec.recognize_beam(h_sd[i][0], lpz_sd[i], recog_args, char_list, rnnlm, strm_idx=i)
             for i in range(self.num_spkrs)]

        if prev:
            self.train()
        return y

    def recognize_batch(self, xs, recog_args, char_list, rnnlm=None):
        """E2E beam search

        :param ndarray xs: input acoustic feature (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        hs = [to_device(self, torch.from_numpy(np.array(xx, dtype=np.float32)))
              for xx in xs]

        # 1. encoder
        xpad = pad_list(hs, 0.0)
        hpad_sd, hlens = self.enc(xpad, ilens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz_sd = [self.ctc.log_softmax(hpad_sd[i]) for i in range(self.num_spkrs)]
        else:
            lpz_sd = None

        # 2. decoder
        y = [self.dec.recognize_beam_batch(hpad_sd[i], hlens, lpz_sd[i], recog_args, char_list, rnnlm, strm_idx=i)
             for i in range(self.num_spkrs)]

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad_sd):
        """E2E attention calculation

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad_sd: batch of padded character id sequence tensor (B, num_spkrs, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            # encoder
            hpad_sd, hlens = self.enc(xs_pad, ilens)

            # Permutation
            ys_pad_sd = ys_pad_sd.transpose(0, 1)  # (num_spkrs, B, Lmax)
            if self.num_spkrs <= 3:
                loss_ctc = torch.stack([self.ctc(hpad_sd[i // self.num_spkrs], hlens, ys_pad_sd[i % self.num_spkrs])
                                        for i in range(self.num_spkrs ** 2)], 1)  # (B, num_spkrs^2)
                loss_ctc, min_perm = self.min_pit_ctc_batch(loss_ctc)
            for i in range(ys_pad_sd.size(1)):  # B
                ys_pad_sd[:, i] = ys_pad_sd[min_perm[i], i]

            # decoder
            att_ws_sd = [self.dec.calculate_all_attentions(hpad_sd[i], hlens, ys_pad_sd[i], strm_idx=i)
                         for i in range(self.num_spkrs)]

        return att_ws_sd


class Encoder(torch.nn.Module):
    """Encoder module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers_sd: number of layers of speaker differentiate part in encoder network
    :param int elayers_rec: number of layers of shared recognition part in encoder network
    :param int eunits: number of lstm units of encoder network
    :param int eprojs: number of projection units of encoder network
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    :param int num_spkrs: number of number of speakers
    """

    def __init__(self, etype, idim, elayers_sd, elayers_rec, eunits, eprojs,
                 subsample, dropout, num_spkrs=2, in_channel=1):
        super(Encoder, self).__init__()
        typ = etype.lstrip("vgg").lstrip("b").rstrip("p")
        if typ != "lstm" and typ != "gru":
            logging.error("Error: need to specify an appropriate encoder architecture")
        if etype.startswith("vgg"):
            if etype[-1] == "p":
                self.enc_mix = torch.nn.ModuleList([VGG2L(in_channel)])
                self.enc_sd = torch.nn.ModuleList([torch.nn.ModuleList([RNNP(get_vgg2l_odim(idim,
                                                                                             in_channel=in_channel),
                                                                              elayers_sd, eunits, eprojs,
                                                                              subsample[:elayers_sd + 1], dropout,
                                                                              typ=typ)])
                                                   for i in range(num_spkrs)])
                self.enc_rec = torch.nn.ModuleList([RNNP(eprojs, elayers_rec, eunits, eprojs,
                                                          subsample[elayers_sd:], dropout, typ=typ)])
                logging.info('Use CNN-VGG + B' + typ.upper() + 'P for encoder')
        else:
            logging.error(
                "Error: need to specify an appropriate encoder architecture")
            sys.exit()

        self.num_spkrs = num_spkrs

    def forward(self, xs_pad, ilens):
        """Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: list: batch of hidden state sequences [num_spkrs x (B, Tmax, eprojs)]
        :rtype: torch.Tensor
        """
        # mixture encoder
        for module in self.enc_mix:
            xs_pad, ilens = module(xs_pad, ilens)

        # SD and Rec encoder
        xs_pad_sd = [xs_pad for i in range(self.num_spkrs)]
        ilens_sd = [ilens for i in range(self.num_spkrs)]
        for ns in range(self.num_spkrs):
            # Encoder_SD: speaker differentiate encoder
            for module in self.enc_sd[ns]:
                xs_pad_sd[ns], ilens_sd[ns] = module(xs_pad_sd[ns], ilens_sd[ns])
            # Encoder_Rec: recognition encoder
            for module in self.enc_rec:
                xs_pad_sd[ns], ilens_sd[ns] = module(xs_pad_sd[ns], ilens_sd[ns])

        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens_sd[0]).unsqueeze(-1))

        return [x.masked_fill(mask, 0.0) for x in xs_pad_sd], ilens_sd[0]


def encoder_for(args, idim, subsample):
    return Encoder(args.etype, idim, args.elayers_sd, args.elayers, args.eunits, args.eprojs, subsample,
                   args.dropout_rate, args.num_spkrs)
