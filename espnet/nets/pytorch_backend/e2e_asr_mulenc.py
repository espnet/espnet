#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import argparse
import logging
import math
import os

import editdistance

import chainer
import numpy as np
import six
import torch

from itertools import groupby

from chainer import reporter

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for

CTC_LOSS_THRESHOLD = 10000


class Reporter(chainer.Chain):
    """A chainer reporter wrapper"""

    def report(self, loss_ctc_list, loss_att, acc, cer_ctc_list, cer, wer, mtl_loss):
        # loss_ctc_list = [weighted CTC, CTC1, CTC2, ... CTCN]
        # cer_ctc_list = [weighted cer_ctc, cer_ctc_1, cer_ctc_2, ... cer_ctc_N]
        num_encs = len(loss_ctc_list) - 1
        reporter.report({'loss_ctc': loss_ctc_list[0]}, self)
        for i in range(num_encs):
            reporter.report({'loss_ctc{}'.format(i+1): loss_ctc_list[i+1]}, self)
        reporter.report({'loss_att': loss_att}, self)
        reporter.report({'acc': acc}, self)
        reporter.report({'cer_ctc': cer_ctc_list[0]}, self)
        for i in range(num_encs):
            reporter.report({'cer_ctc{}'.format(i+1): cer_ctc_list[i+1]}, self)
        reporter.report({'cer': cer}, self)
        reporter.report({'wer': wer}, self)
        logging.info('mtl loss:' + str(mtl_loss))
        reporter.report({'loss': mtl_loss}, self)


class E2E(ASRInterface, torch.nn.Module):
    """E2E module

    :param List idims: List of dimensions of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    def __init__(self, idims, odim, args):
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.space = args.sym_space
        self.blank = args.sym_blank
        self.reporter = Reporter()
        self.num_encs = args.num_encs
        self.share_ctc = args.share_ctc

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        self.subsample_list = []
        for idx in range(self.num_encs):
            # +1 means input (+1) and layers outputs (args.elayer)
            subsample = np.ones(args.elayers[idx] + 1, dtype=np.int)
            if args.etype[idx].endswith("p") and not args.etype[idx].startswith("vgg"):
                ss = args.subsample[idx].split("_")
                for j in range(min(args.elayers[idx] + 1, len(ss))):
                    subsample[j] = int(ss[j])
            else:
                logging.warning(
                    'Encoder {}: Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.'.format(idx+1))
            logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
            self.subsample_list.append(subsample)

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # speech translation related
        self.replace_sos = args.replace_sos

        self.frontend = None

        # encoder
        self.enc = encoder_for(args, idims, self.subsample_list)
        # ctc
        self.ctc = ctc_for(args, odim)
        # attention
        self.att = att_for(args)
        # hierarchical attention network
        han = att_for(args, han_mode=True)
        self.att.append(han)
        # decoder
        self.dec = decoder_for(args, odim, self.sos, self.eos, self.att, labeldist)

        if args.mtlalpha > 0 is not None and self.num_encs > 1:
            # weights-ctc, e.g. ctc_loss = w_1*ctc_1_loss + w_2 * ctc_2_loss + w_N * ctc_N_loss
            self.weights_ctc_train = args.weights_ctc_train / np.sum(args.weights_ctc_train) # normalize
            self.weights_ctc_dec = args.weights_ctc_dec / np.sum(args.weights_ctc_dec) # normalize
            logging.info('ctc weights (training during training): ' + ' '.join([str(x) for x in self.weights_ctc_train]))
            logging.info('ctc weights (decoding during training): ' + ' '.join([str(x) for x in self.weights_ctc_dec]))
        else:
            self.weights_ctc_dec = [1.0]
            self.weights_ctc_train = [1.0]


        # weight initialization
        self.init_like_chainer()

        # options for beam search
        if args.report_cer or args.report_wer:
            recog_args = {'beam_size': args.beam_size, 'penalty': args.penalty,
                          'ctc_weight': args.ctc_weight, 'maxlenratio': args.maxlenratio,
                          'minlenratio': args.minlenratio, 'lm_weight': args.lm_weight,
                          'rnnlm': args.rnnlm, 'nbest': args.nbest,
                          'space': args.sym_space, 'blank': args.sym_blank,
                          'tgt_lang': False, 'ctc_weights_dec': self.weights_ctc_dec}

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
                elif data.dim() in (3, 4):
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

    def forward(self, xs_pad_list, ilens_list, ys_pad):
        """E2E forward

        :param List xs_pad_list: list of batch (torch.Tensor) of padded input sequences [(B, Tmax_1, idim), (B, Tmax_2, idim),..]
        :param List ilens_list: list of batch (torch.Tensor) of lengths of input sequences [(B), (B), ..]
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        if self.replace_sos:
            tgt_lang_ids = ys_pad[:, 0:1]
            ys_pad = ys_pad[:, 1:]  # remove target language ID in the beginning
        else:
            tgt_lang_ids = None

        hs_pad_list, hlens_list, self.loss_ctc_list = [], [], []
        for idx in range(self.num_encs):
            # 1. Encoder
            hs_pad, hlens, _ = self.enc[idx](xs_pad_list[idx], ilens_list[idx])

            # 2. CTC loss
            if self.mtlalpha == 0:
                self.loss_ctc_list.append(None)
            else:
                ctc_idx = 0 if self.share_ctc else idx
                loss_ctc = self.ctc[ctc_idx](hs_pad, hlens, ys_pad)
                self.loss_ctc_list.append(loss_ctc)
            hs_pad_list.append(hs_pad)
            hlens_list.append(hlens)

        # 3. attention loss
        if self.mtlalpha == 1:
            self.loss_att, acc = None, None
        else:
            self.loss_att, acc, _ = self.dec(hs_pad_list, hlens_list, ys_pad, tgt_lang_ids=tgt_lang_ids)
        self.acc = acc

        # 4. compute cer without beam search
        if self.mtlalpha == 0:
            cer_ctc_list = [None] * (self.num_encs + 1)
        else:
            cer_ctc_list = []
            for ind in range(self.num_encs):
                cers = []
                ctc_idx = 0 if self.share_ctc else ind
                y_hats = self.ctc[ctc_idx].argmax(hs_pad_list[ind]).data
                for i, y in enumerate(y_hats):
                    y_hat = [x[0] for x in groupby(y)]
                    y_true = ys_pad[i]

                    seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                    seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                    seq_hat_text = "".join(seq_hat).replace(self.space, ' ')
                    seq_hat_text = seq_hat_text.replace(self.blank, '')
                    seq_true_text = "".join(seq_true).replace(self.space, ' ')

                    hyp_chars = seq_hat_text.replace(' ', '')
                    ref_chars = seq_true_text.replace(' ', '')
                    if len(ref_chars) > 0:
                        cers.append(editdistance.eval(hyp_chars, ref_chars) / len(ref_chars))

                cer_ctc = sum(cers) / len(cers) if cers else None
                cer_ctc_list.append(cer_ctc)
            cer_ctc_weighted =  np.sum([item * self.weights_ctc_train[i] for i, item in enumerate(cer_ctc_list)])
            cer_ctc_list = [float(cer_ctc_weighted)] + [float(item) for item in cer_ctc_list]

        # 5. compute cer/wer
        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz_list = []
                for idx in range(self.num_encs):
                    ctc_idx = 0 if self.share_ctc else idx
                    lpz = self.ctc[ctc_idx].log_softmax(hs_pad_list[idx]).data
                    lpz_list.append(lpz)
            else:
                lpz_list = None

            word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []
            nbest_hyps = self.dec.recognize_beam_batch(
                hs_pad_list, hlens_list, lpz_list,
                self.recog_args, self.char_list,
                self.rnnlm,
                tgt_lang_ids=tgt_lang_ids.squeeze(1).tolist() if self.replace_sos else None)
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.recog_args.blank, '')
                seq_true_text = "".join(seq_true).replace(self.recog_args.space, ' ')

                hyp_words = seq_hat_text.split()
                ref_words = seq_true_text.split()
                word_eds.append(editdistance.eval(hyp_words, ref_words))
                word_ref_lens.append(len(ref_words))
                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                char_eds.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))

            wer = 0.0 if not self.report_wer else float(sum(word_eds)) / sum(word_ref_lens)
            cer = 0.0 if not self.report_cer else float(sum(char_eds)) / sum(char_ref_lens)

        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = self.loss_att
            loss_att_data = float(self.loss_att)
            loss_ctc_data_list = [None] * (self.num_encs + 1)
        elif alpha == 1:
            self.loss = torch.sum(torch.cat([item * self.weights_ctc_train[i] for i, item in enumerate(self.loss_ctc_list)]))
            loss_att_data = None
            loss_ctc_data_list = [float(self.loss)] + [float(item) for item in self.loss_ctc_list]
        else:
            self.loss_ctc = torch.sum(torch.cat([item * self.weights_ctc_train[i] for i, item in enumerate(self.loss_ctc_list)]))
            self.loss = alpha * self.loss_ctc + (1 - alpha) * self.loss_att
            loss_att_data = float(self.loss_att)
            loss_ctc_data_list = [float(self.loss_ctc)] + [float(item) for item in self.loss_ctc_list]

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data_list, loss_att_data, acc, cer_ctc_list, cer, wer, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def recognize(self, x_list, recog_args, char_list, rnnlm=None):
        """E2E beam search

        :param list of ndarray x: list of input acoustic feature [(T1, D), (T2,D),...]
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        ilens_list = [[x_list[idx].shape[0]] for idx in range(self.num_encs)]

        # subsample frame
        x_list = [x_list[idx][::self.subsample_list[idx][0], :] for idx in range(self.num_encs)]
        x_list = [to_device(self, to_torch_tensor(x_list[idx]).float()) for idx in range(self.num_encs)]
        # make a utt list (1) to use the same interface for encoder
        xs_list = [x_list[idx].contiguous().unsqueeze(0) for idx in range(self.num_encs)]

        # 1. encoder
        hs_list, hlens_list = [], []
        for idx in range(self.num_encs):
            hs, _, _ = self.enc[idx](xs_list[idx], ilens_list[idx])
            hs_list.append(hs[0])

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            if self.share_ctc:
                lpz_list = [self.ctc[0].log_softmax(hs_list[idx].unsqueeze(0))[0] for idx in range(self.num_encs)]
            else:
                lpz_list = [self.ctc[idx].log_softmax(hs_list[idx].unsqueeze(0))[0] for idx in range(self.num_encs)]
        else:
            lpz_list = None

        # 2. Decoder
        # decode the first utterance
        y = self.dec.recognize_beam(hs_list, lpz_list, recog_args, char_list, rnnlm)

        if prev:
            self.train()

        return y

    def recognize_batch(self, xs_list, recog_args, char_list, rnnlm=None):
        """E2E beam search

        :param list xs_list: list of list of input acoustic feature arrays [[(T1_1, D), (T1_2, D), ...],[(T2_1, D), (T2_2, D), ...], ...]
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        ilens_list = [np.fromiter((xx.shape[0] for xx in xs_list[idx]), dtype=np.int64) for idx in range(self.num_encs)]

        # subsample frame
        xs_list = [[xx[::self.subsample_list[idx][0], :] for xx in xs_list[idx]] for idx in range(self.num_encs)]

        xs_list = [[to_device(self, to_torch_tensor(xx).float()) for xx in xs_list[idx]] for idx in range(self.num_encs)]
        xs_pad_list = [pad_list(xs_list[idx], 0.0) for idx in range(self.num_encs)]

        # 1. Encoder
        hs_pad_list, hlens_list = [], []
        for idx in range(self.num_encs):
            hs_pad, hlens, _ = self.enc[idx](xs_pad_list[idx], ilens_list[idx])
            hs_pad_list.append(hs_pad)
            hlens_list.append(hlens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            if self.share_ctc:
                lpz_list = [self.ctc[0].log_softmax(hs_pad_list[idx]) for idx in range(self.num_encs)]
            else:
                lpz_list = [self.ctc[idx].log_softmax(hs_pad_list[idx]) for idx in range(self.num_encs)]
        else:
            lpz_list = None

        # 2. Decoder
        hlens_list = [torch.tensor(list(map(int, hlens_list[idx]))) for idx in range(self.num_encs)]  # make sure hlens is tensor
        y = self.dec.recognize_beam_batch(hs_pad_list, hlens_list, lpz_list, recog_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad_list, ilens_list, ys_pad):
        """E2E attention calculation

        :param List xs_pad_list: list of batch (torch.Tensor) of padded input sequences [(B, Tmax_1, idim), (B, Tmax_2, idim),..]
        :param List ilens_list: list of batch (torch.Tensor) of lengths of input sequences [(B), (B), ..]
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) multi-encoder case => [(B, Lmax, Tmax1), (B, Lmax, Tmax2), ..., (B, Lmax, NumEncs)]
            3) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray or list
        """
        with torch.no_grad():
            # 1. Encoder
            if self.replace_sos:
                tgt_lang_ids = ys_pad[:, 0:1]
                ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining
            else:
                tgt_lang_ids = None

            hs_pad_list, hlens_list = [], []
            for idx in range(self.num_encs):
                hs_pad, hlens, _ = self.enc[idx](xs_pad_list[idx], ilens_list[idx])
                hs_pad_list.append(hs_pad)
                hlens_list.append(hlens)

            # 2. Decoder
            att_ws = self.dec.calculate_all_attentions(hs_pad_list, hlens_list, ys_pad, tgt_lang_ids=tgt_lang_ids)

        return att_ws
