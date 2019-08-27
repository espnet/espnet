#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import argparse
import logging
import math
import sys
import os

import editdistance

import chainer
import numpy as np
import six
import torch

from itertools import groupby

from chainer import reporter

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import Encoder
from espnet.nets.pytorch_backend.rnn.encoders import RNNP
from espnet.nets.pytorch_backend.rnn.encoders import VGG2L

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


class PIT(object):
    """Permutation Invariant Training (PIT) module

    :parameter int num_spkrs: number of speakers for PIT process (2 or 3)
    """

    def __init__(self, num_spkrs):
        self.num_spkrs = num_spkrs
        if self.num_spkrs == 2:
            self.perm_choices = [[0, 1], [1, 0]]
        elif self.num_spkrs == 3:
            self.perm_choices = [[0, 1, 2], [0, 2, 1], [1, 2, 0], [1, 0, 2], [2, 0, 1], [2, 1, 0]]
        else:
            raise ValueError

    def min_pit_sample(self, loss):
        """PIT min_pit_sample

        :param 1-D torch.Tensor loss: list of losses for one sample,
            including [h1r1, h1r2, h2r1, h2r2] or [h1r1, h1r2, h1r3, h2r1, h2r2, h2r3, h3r1, h3r2, h3r3]
        :return min_loss
        :rtype torch.Tensor (1)
        :return permutation
        :rtype List: len=2
        """

        if self.num_spkrs == 2:
            score_perms = torch.stack([loss[0] + loss[3],
                                       loss[1] + loss[2]]) / self.num_spkrs
        elif self.num_spkrs == 3:
            score_perms = torch.stack([loss[0] + loss[4] + loss[8],
                                       loss[0] + loss[5] + loss[7],
                                       loss[1] + loss[5] + loss[6],
                                       loss[1] + loss[3] + loss[8],
                                       loss[2] + loss[3] + loss[7],
                                       loss[2] + loss[4] + loss[6]]) / self.num_spkrs

        perm_loss, min_idx = torch.min(score_perms, 0)
        permutation = self.perm_choices[min_idx]

        return perm_loss, permutation

    def pit_process(self, losses):
        """PIT pit_process

        :param torch.Tensor losses: losses (B, 1|4|9)
        :return pit_loss
        :rtype torch.Tensor (B)
        :return permutation
        :rtype torch.LongTensor (B, 1|2|3)
        """

        bs = losses.size(0)
        ret = [self.min_pit_sample(losses[i]) for i in range(bs)]

        loss_perm = torch.stack([r[0] for r in ret], dim=0).to(losses.device)  # (B)
        permutation = torch.tensor([r[1] for r in ret]).long().to(losses.device)

        return torch.mean(loss_perm), permutation


class E2E(ASRInterface, torch.nn.Module):
    """E2E module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    """

    def __init__(self, idim, odim, args):
        torch.nn.Module.__init__(self)
        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.reporter = Reporter()
        self.num_spkrs = args.num_spkrs
        self.spa = args.spa
        self.pit = PIT(self.num_spkrs)

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
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type,
            transcript=args.train_json)
        else:
            labeldist = None

        if getattr(args, "use_frontend", False):  # use getattr to keep compatibility
            # Relative importing because of using python3 syntax
            from espnet.nets.pytorch_backend.frontends.feature_transform \
                import feature_transform_for
            from espnet.nets.pytorch_backend.frontends.frontend \
                import frontend_for

            self.frontend = frontend_for(args, idim)
            self.feature_transform = feature_transform_for(args, (idim - 1) * 2)
            idim = args.n_mels
        else:
            self.frontend = None

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

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, num_spkrs, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 0. Frontend
        if self.frontend is not None:
            hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
            if isinstance(hs_pad, list):
                hlens_n = [None] * self.num_spkrs
                for i in range(self.num_spkrs):
                    hs_pad[i], hlens_n[i] = self.feature_transform(hs_pad[i], hlens)
                hlens = hlens_n
            else:
                hs_pad, hlens = self.feature_transform(hs_pad, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens

        # 1. Encoder
        if not isinstance(hs_pad, list):  # single-channel input xs_pad (single- or multi-speaker)
            hs_pad, hlens, _ = self.enc(hs_pad, hlens)
        else:  # multi-channel multi-speaker input xs_pad
            for i in range(self.num_spkrs):
                hs_pad[i], hlens[i], _ = self.enc(hs_pad[i], hlens[i])

        # 2. CTC loss
        if self.mtlalpha == 0:
            loss_ctc, min_perm = None, None
        else:
            if not isinstance(hs_pad, list):  # single-speaker input xs_pad
                loss_ctc = torch.mean(self.ctc(hs_pad, hlens, ys_pad))
            else:  # multi-speaker input xs_pad
                ys_pad = ys_pad.transpose(0, 1)  # (num_spkrs, B, Lmax)
                loss_ctc_perm = torch.stack([self.ctc(hs_pad[i // self.num_spkrs],
                                                      hlens[i // self.num_spkrs],
                                                      ys_pad[i % self.num_spkrs])
                                             for i in range(self.num_spkrs ** 2)], dim=1)  # (B, num_spkrs^2)
                loss_ctc, min_perm = self.pit.pit_process(loss_ctc_perm)
                logging.info('ctc loss:' + str(float(loss_ctc)))

        # 3. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
        else:
            if not isinstance(hs_pad, list):  # single-speaker input xs_pad
                loss_att, acc, _ = self.dec(hs_pad, hlens, ys_pad)
            else:
                for i in range(ys_pad.size(1)):  # B
                    ys_pad[:, i] = ys_pad[min_perm[i], i]
                rslt = [self.dec(hs_pad[i], hlens[i], ys_pad[i], strm_idx=i) for i in range(self.num_spkrs)]
                loss_att = sum([r[0] for r in rslt]) / float(len(rslt))
                acc = sum([r[1] for r in rslt]) / float(len(rslt))
        self.acc = acc

        # 5. compute cer/wer
        if self.training or not (self.report_cer or self.report_wer) or not isinstance(hs_pad, list):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz = [self.ctc.log_softmax(hs_pad[i]).data for i in range(self.num_spkrs)]
            else:
                lpz = None

            word_eds, char_eds, word_ref_lens, char_ref_lens = [], [], [], []
            nbest_hyps = [self.dec.recognize_beam_batch(hs_pad[i], torch.tensor(hlens[i]), lpz[i],
                                                        self.recog_args, self.char_list, self.rnnlm, strm_idx=i)
                          for i in range(self.num_spkrs)]
            # remove <sos> and <eos>
            y_hats = [[nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps[i]] for i in range(self.num_spkrs)]
            for i in range(len(y_hats[0])):
                hyp_words = []
                hyp_chars = []
                ref_words = []
                ref_chars = []
                for ns in range(self.num_spkrs):
                    y_hat = y_hats[ns][i]
                    y_true = ys_pad[ns][i]

                    seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                    seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                    seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                    seq_hat_text = seq_hat_text.replace(self.recog_args.blank, '')
                    seq_true_text = "".join(seq_true).replace(self.recog_args.space, ' ')

                    hyp_words.append(seq_hat_text.split())
                    ref_words.append(seq_true_text.split())
                    hyp_chars.append(seq_hat_text.replace(' ', ''))
                    ref_chars.append(seq_true_text.replace(' ', ''))

                tmp_word_ed = [editdistance.eval(hyp_words[ns // self.num_spkrs], ref_words[ns % self.num_spkrs])
                               for ns in range(self.num_spkrs ** 2)]  # h1r1,h1r2,h2r1,h2r2
                tmp_char_ed = [editdistance.eval(hyp_chars[ns // self.num_spkrs], ref_chars[ns % self.num_spkrs])
                               for ns in range(self.num_spkrs ** 2)]  # h1r1,h1r2,h2r1,h2r2

                word_eds.append(self.pit.min_pit_sample(torch.tensor(tmp_word_ed))[0])
                word_ref_lens.append(len(sum(ref_words, [])))
                char_eds.append(self.pit.min_pit_sample(torch.tensor(tmp_char_ed))[0])
                char_ref_lens.append(len(''.join(ref_chars)))

            wer = 0.0 if not self.report_wer else float(sum(word_eds)) / sum(word_ref_lens)
            cer = 0.0 if not self.report_cer else float(sum(char_eds)) / sum(char_ref_lens)

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
        return self.loss

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
        ilens = [x.shape[0]]

        # subsample frame
        x = x[::self.subsample[0], :]
        h = to_device(self, to_torch_tensor(x).float())
        # make a utt list (1) to use the same interface for encoder
        hs = h.continguous().unsqueeze(0)

        # 0. Frontend
        if self.frontend is not None:
            hs, hlens, mask = self.frontend(hs, ilens)
            hlens_n = [None] * self.num_spkrs
            for i in range(self.num_spkrs):
                hs[i], hlens_n[i] = self.feature_transform(hs[i], hlens)
            hlens = hlens_n
        else:
            hs, hlens = hs, ilens

        # 1. Encoder
        if not isinstance(hs, list):  # single-channel multi-speaker input x
            hs, hlens, _ = self.enc(hs, hlens)
        else:  # multi-channel multi-speaker input x
            for i in range(self.num_spkrs):
                hs[i], hlens[i], _ = self.enc(hs[i], hlens[i])

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = [self.ctc.log_softmax(i)[0] for i in hs]
        else:
            lpz = None

        # 2. decoder
        # decode the first utterance
        y = [self.dec.recognize_beam(hs[i][0], lpz[i], recog_args, char_list, rnnlm, strm_idx=i)
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
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)

        # 0. Frontend
        if self.frontend is not None:
            hs_pad, hlens, mask = self.frontend(xs_pad, ilens)
            hlens_n = [None] * self.num_spkrs
            for i in range(self.num_spkrs):
                hs_pad[i], hlens_n[i] = self.feature_transform(hs_pad[i], hlens)
            hlens = hlens_n
        else:
            hs_pad, hlens = xs_pad, ilens

        # 1. Encoder
        if not isinstance(hs_pad, list):  # single-channel multi-speaker input x
            hs_pad, hlens, _ = self.enc(hs_pad, hlens)
        else:  # multi-channel multi-speaker input x
            for i in range(self.num_spkrs):
                hs_pad[i], hlens[i], _ = self.enc(hs_pad[i], hlens[i])


        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = [self.ctc.log_softmax(hs_pad[i]) for i in range(self.num_spkrs)]
        else:
            lpz = None

        # 2. decoder
        y = [self.dec.recognize_beam_batch(hs_pad[i], hlens[i], lpz[i], recog_args, char_list, rnnlm, strm_idx=i)
             for i in range(self.num_spkrs)]

        if prev:
            self.train()
        return y

    def enhance(self, xs):
        """Forwarding only the frontend stage

        :param ndarray xs: input acoustic feature (T, C, F)
        """

        if self.frontend is None:
            raise RuntimeError('Frontend doesn\'t exist')
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)
        enhanced, hlensm, mask = self.frontend(xs_pad, ilens)
        if prev:
            self.train()
        
        if isinstance(enhanced, (tuple, list)):
            enhanced = list(enhanced)
            masks = list(masks)
            for idx in range(len(enhanced)):  # number of speakers
                enhanced[idx] = enhanced[idx].cpu().numpy()
                mask[idx] = mask[idx].cpu().numpy()
            return enhanced, mask, ilens
        return enhanced.cpu().numpy(), mask.cpu().numpy(), ilens

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, num_spkrs, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            # 0. Frontend
            if self.frontend is not None:
                hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
                hlens_n = [None] * self.num_spkrs
                for i in range(self.num_spkrs):
                    hs_pad[i], hlens_n[i] = self.feature_transform(hs_pad[i], hlens)
                hlens = hlens_n
            else:
                hs_pad, hlens = xs_pad, ilens

            # 1. Encoder
            if not isinstance(hs_pad, list):  # single-channel multi-speaker input x
                hs_pad, hlens, _ = self.enc(hs_pad, hlens)
            else:  # multi-channel multi-speaker input x
                for i in range(self.num_spkrs):
                    hs_pad[i], hlens[i], _ = self.enc(hs_pad[i], hlens[i])

            # Permutation
            ys_pad = ys_pad.transpose(0, 1)  # (num_spkrs, B, Lmax)
            if self.num_spkrs <= 3:
                loss_ctc = torch.stack([self.ctc(hs_pad[i // self.num_spkrs],
                                                 hlens[i // self.num_spkrs],
                                                 ys_pad[i % self.num_spkrs])
                                        for i in range(self.num_spkrs ** 2)], 1)  # (B, num_spkrs^2)
                loss_ctc, min_perm = self.pit.pit_process(loss_ctc)
            for i in range(ys_pad.size(1)):  # B
                ys_pad[:, i] = ys_pad[min_perm[i], i]

            # 2. Decoder
            att_ws = [self.dec.calculate_all_attentions(hs_pad[i], hlens[i], ys_pad[i], strm_idx=i)
                         for i in range(self.num_spkrs)]

        return att_ws


class EncoderMix(torch.nn.Module):
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
        super(EncoderMix, self).__init__()
        typ = etype.lstrip("vgg").rstrip("p")
        if typ not in ['lstm', 'gru', 'blstm', 'bgru']:
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
            xs_pad, ilens, _ = module(xs_pad, ilens)

        # SD and Rec encoder
        xs_pad_sd = [xs_pad for i in range(self.num_spkrs)]
        ilens_sd = [ilens for i in range(self.num_spkrs)]
        for ns in range(self.num_spkrs):
            # Encoder_SD: speaker differentiate encoder
            for module in self.enc_sd[ns]:
                xs_pad_sd[ns], ilens_sd[ns], _ = module(xs_pad_sd[ns], ilens_sd[ns])
            # Encoder_Rec: recognition encoder
            for module in self.enc_rec:
                xs_pad_sd[ns], ilens_sd[ns], _ = module(xs_pad_sd[ns], ilens_sd[ns])

        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens_sd[0]).unsqueeze(-1))

        return [x.masked_fill(mask, 0.0) for x in xs_pad_sd], ilens_sd, None


def encoder_for(args, idim, subsample):
    if getattr(args, "use_frontend", False):  # use getattr to keep compatibility
        # with frontend, the mixed speech are separated as streams for each speaker
        from espnet.nets.pytorch_backend.rnn.encoders import encoder_for as encoder_for_single
        return encoder_for_single(args, idim, subsample)
    else:
        return EncoderMix(args.etype, idim, args.elayers_sd, args.elayers, args.eunits, args.eprojs, subsample,
                          args.dropout_rate, args.num_spkrs)
