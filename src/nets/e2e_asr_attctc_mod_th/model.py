#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging
import math
import os
import sys

import chainer
import numpy as np
import six
import torch

from chainer import reporter
from torch.autograd import Variable

from e2e_asr_common import label_smoothing_dist

from e2e_asr_attctc_mod_th.attention_th import AttAdd
from e2e_asr_attctc_mod_th.attention_th import AttCov
from e2e_asr_attctc_mod_th.attention_th import AttCovLoc
from e2e_asr_attctc_mod_th.attention_th import AttDot
from e2e_asr_attctc_mod_th.attention_th import AttLoc
from e2e_asr_attctc_mod_th.attention_th import AttLoc2D
from e2e_asr_attctc_mod_th.attention_th import AttLocRec
from e2e_asr_attctc_mod_th.attention_th import AttMultiHeadAdd
from e2e_asr_attctc_mod_th.attention_th import AttMultiHeadDot
from e2e_asr_attctc_mod_th.attention_th import AttMultiHeadLoc
from e2e_asr_attctc_mod_th.attention_th import AttMultiHeadMultiResLoc
from e2e_asr_attctc_mod_th.attention_th import NoAtt
from e2e_asr_attctc_mod_th.ctc_th import CTC
from e2e_asr_attctc_mod_th.decoder_th import Decoder
from e2e_asr_attctc_mod_th.encoder_th import Encoder

torch_is_old = torch.__version__.startswith("0.3.")

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5


def to_cuda(m, x):
    assert isinstance(m, torch.nn.Module)
    device_id = torch.cuda.device_of(next(m.parameters()).data).idx
    if device_id == -1:
        return x
    return x.cuda(device_id)


def lecun_normal_init_parameters(module):
    for n, p in module.named_parameters():
        data = p.data

        # Skip pre-trained RNNLM
        if 'rnnlm_cf' in n:
            continue

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


def linear_tensor(linear, x):
    '''Apply linear matrix operation only for the last dimension of a tensor

    :param Link linear: Linear link (M x N matrix)
    :param Variable x: Tensor (D_1 x D_2 x ... x M matrix)
    :return:
    :param Variable x: Tensor (D_1 x D_2 x ... x N matrix)
    '''
    y = linear(x.contiguous().view((-1, x.size()[-1])))
    return y.view((x.size()[:-1] + (-1,)))


class Reporter(chainer.Chain):
    def report(self, loss_ctc, loss_att, acc, mtl_loss):
        reporter.report({'loss_ctc': loss_ctc}, self)
        reporter.report({'loss_att': loss_att}, self)
        reporter.report({'acc': acc}, self)
        logging.info('mtl loss:' + str(mtl_loss))
        reporter.report({'loss': mtl_loss}, self)


# TODO(watanabe) merge Loss and E2E: there is no need to make these separately
class Loss(torch.nn.Module):
    def __init__(self, predictor, mtlalpha):
        super(Loss, self).__init__()
        self.mtlalpha = mtlalpha
        self.loss = None
        self.accuracy = None
        self.predictor = predictor
        self.reporter = Reporter()

    def forward(self, x):
        '''Loss forward

        :param x:
        :return:
        '''
        self.loss = None
        loss_ctc, loss_att, acc = self.predictor(x)
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = loss_att.data[0] if torch_is_old else float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = loss_ctc.data[0] if torch_is_old else float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = loss_att.data[0] if torch_is_old else float(loss_att)
            loss_ctc_data = loss_ctc.data[0] if torch_is_old else float(loss_ctc)

        loss_data = self.loss.data[0] if torch_is_old else float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data, loss_att_data, acc, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', self.loss.data)

        return self.loss


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    if torch_is_old:
        if isinstance(xs[0], Variable):
            new = xs[0].data.new
            v = xs[0].volatile
        else:
            new = xs[0].new
            v = False
        pad = Variable(
            new(n_batch, max_len, * xs[0].size()[1:]).zero_() + pad_value,
            volatile=v)
    else:
        pad = xs[0].data.new(
            n_batch, max_len, * xs[0].size()[1:]).zero_() + pad_value

    for i in six.moves.range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def set_forget_bias_to_one(bias):
    n = bias.size(0)
    start, end = n // 4, n // 2
    bias.data[start:end].fill_(1.)


class E2E(torch.nn.Module):
    def __init__(self, idim, odim, args, rnnlm_cf, rnnlm_init):
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
            for j in six.moves.range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args.lsm_type and args.lsm_weight > 0 and os.path.isfile(args.train_json):
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
                           labeldist, args.lsm_weight, args.gen_feat,
                           rnnlm_cf, args.cf_type,
                           rnnlm_init, args.lm_loss_weight, args.internal_lm,
                           args.share_softmax)
        # TODO(hirofumi): add dropout

        # weight initialization
        self.init_like_chainer()
        # additional forget-bias init in encoder ?
        # for m in self.modules():
        #     if isinstance(m, torch.nn.LSTM):
        #         for name, p in m.named_parameters():
        #             if "bias_ih" in name:
        #                 set_forget_bias_to_one(p)

    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        lecun_normal_init_parameters(self)

        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

        # Initialize decoder with pre-trained RNNLM
        if self.dec.rnnlm_init is not None:
            logging.info('Initialize the decoder with pre-trained RNNLM')
            # LSTM
            self.dec.decoder_lm.weight_ih.data = self.dec.rnnlm_init.predictor.l1.weight_ih.data
            self.dec.decoder_lm.weight_hh.data = self.dec.rnnlm_init.predictor.l1.weight_hh.data
            self.dec.decoder_lm.bias_ih.data = self.dec.rnnlm_init.predictor.l1.bias_ih.data
            self.dec.decoder_lm.bias_hh.data = self.dec.rnnlm_init.predictor.l1.bias_hh.data
            # embedding
            self.dec.embed.weight.data = self.dec.rnnlm_init.predictor.embed.weight.data
            # softmax
            self.dec.output.weight.data = self.dec.rnnlm_init.predictor.lo.weight.data
            self.dec.output.bias.data = self.dec.rnnlm_init.predictor.lo.bias.data
            if self.dec.lm_loss_weight > 0 and not self.dec.share_softmax:
                self.dec.rnnlm_lo.weight.data = self.dec.rnnlm_init.predictor.lo.weight.data
                self.dec.rnnlm_lo.bias.data = self.dec.rnnlm_init.predictor.lo.bias.data

        # Initialize bias in the gating part with -1
        if self.dec.cf_type:
            self.dec.fc_lm_gate.bias.data.fill_(-1)

    # x[i]: ('utt_id', {'ilen':'xxx',...}})
    def forward(self, data):
        '''E2E forward

        :param data:
        :return:
        '''
        # utt list of frame x dim
        xs = [d[1]['feat'] for d in data]
        # remove 0-output-length utterances
        tids = [d[1]['output'][0]['tokenid'].split() for d in data]
        filtered_index = filter(lambda i: len(tids[i]) > 0, six.moves.range(len(xs)))
        sorted_index = sorted(filtered_index, key=lambda i: -len(xs[i]))
        if len(sorted_index) != len(xs):
            logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                len(xs), len(sorted_index)))
        xs = [xs[i] for i in sorted_index]
        # utt list of olen
        ys = [np.fromiter(map(int, tids[i]), dtype=np.int64)
              for i in sorted_index]
        if torch_is_old:
            ys = [to_cuda(self, Variable(torch.from_numpy(y), volatile=not self.training)) for y in ys]
        else:
            ys = [to_cuda(self, torch.from_numpy(y)) for y in ys]

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        if torch_is_old:
            hs = [to_cuda(self, Variable(torch.from_numpy(xx), volatile=not self.training)) for xx in xs]
        else:
            hs = [to_cuda(self, torch.from_numpy(xx)) for xx in xs]

        # 1. encoder
        xpad = pad_list(hs, 0.0)
        hpad, hlens = self.enc(xpad, ilens)

        # # 3. CTC loss
        if self.mtlalpha == 0:
            loss_ctc = None
        else:
            loss_ctc = self.ctc(hpad, hlens, ys)

        # 4. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
        else:
            loss_att, acc = self.dec(hpad, hlens, ys)

        return loss_ctc, loss_att, acc

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        '''E2E beam search

        :param x:
        :param recog_args:
        :param char_list:
        :return:
        '''
        prev = self.training
        self.eval()
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        if torch_is_old:
            h = to_cuda(self, Variable(torch.from_numpy(
                np.array(x, dtype=np.float32)), volatile=True))
        else:
            h = to_cuda(self, torch.from_numpy(
                np.array(x, dtype=np.float32)))

        # 1. encoder
        # make a utt list (1) to use the same interface for encoder
        h, _ = self.enc(h.unsqueeze(0), ilen)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(h).data[0]
        else:
            lpz = None

        # 2. decoder
        # decode the first utterance
        y = self.dec.recognize_beam(h[0], lpz, recog_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, data):
        '''E2E attention calculation

        :param list data: list of dicts of the input (B)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
         :rtype: float ndarray
        '''
        if not torch_is_old:
            torch.set_grad_enabled(False)

        # utt list of frame x dim
        xs = [d[1]['feat'] for d in data]

        # remove 0-output-length utterances
        tids = [d[1]['output'][0]['tokenid'].split() for d in data]
        filtered_index = filter(lambda i: len(tids[i]) > 0, range(len(xs)))
        sorted_index = sorted(filtered_index, key=lambda i: -len(xs[i]))
        if len(sorted_index) != len(xs):
            logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                len(xs), len(sorted_index)))
        xs = [xs[i] for i in sorted_index]

        # utt list of olen
        ys = [np.fromiter(map(int, tids[i]), dtype=np.int64)
              for i in sorted_index]
        if torch_is_old:
            ys = [to_cuda(self, Variable(torch.from_numpy(y), volatile=True)) for y in ys]
        else:
            ys = [to_cuda(self, torch.from_numpy(y)) for y in ys]

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        if torch_is_old:
            hs = [to_cuda(self, Variable(torch.from_numpy(xx), volatile=True)) for xx in xs]
        else:
            hs = [to_cuda(self, torch.from_numpy(xx)) for xx in xs]

        # encoder
        xpad = pad_list(hs, 0.0)
        hpad, hlens = self.enc(xpad, ilens)

        # decoder
        att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys)

        if not torch_is_old:
            torch.set_grad_enabled(True)

        return att_ws


def mask_by_length(xs, length, fill=0):
    assert xs.size(0) == len(length)
    ret = Variable(xs.data.new(*xs.size()).fill_(fill))
    for i, l in enumerate(length):
        ret[i, :l] = xs[i, :l]
    return ret


def th_accuracy(y_all, pad_target, ignore_label):
    pad_pred = y_all.data.view(pad_target.size(
        0), pad_target.size(1), y_all.size(1)).max(2)[1]
    mask = pad_target.data != ignore_label
    numerator = torch.sum(pad_pred.masked_select(
        mask) == pad_target.data.masked_select(mask))
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


def calculate_all_attentions(self, hpad, hlen, ys):
    '''Calculate all of attentions

    :return: numpy array format attentions
    '''
    hlen = list(map(int, hlen))
    hpad = mask_by_length(hpad, hlen, 0)
    self.loss = None
    # prepare input and output word sequences with sos/eos IDs
    eos = Variable(ys[0].data.new([self.eos]))
    sos = Variable(ys[0].data.new([self.sos]))
    ys_in = [torch.cat([sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, eos], dim=0) for y in ys]

    # padding for ys with -1
    # pys: utt x olen
    pad_ys_in = pad_list(ys_in, self.eos)
    pad_ys_out = pad_list(ys_out, self.ignore_id)

    # get length info
    olength = pad_ys_out.size(1)

    # initialization
    c_list = [self.zero_state(hpad)]
    z_list = [self.zero_state(hpad)]
    for l in six.moves.range(1, self.dlayers):
        c_list.append(self.zero_state(hpad))
        z_list.append(self.zero_state(hpad))
    att_w = None
    att_ws = []
    self.att.reset()  # reset pre-computation of h

    # pre-computation of embedding
    eys = self.embed(pad_ys_in)  # utt x olen x zdim

    # loop for an output sequence
    for i in six.moves.range(olength):
        att_c, att_w = self.att(hpad, hlen, z_list[0], att_w)
        ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
        z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
        for l in six.moves.range(1, self.dlayers):
            z_list[l], c_list[l] = self.decoder[l](
                z_list[l - 1], (z_list[l], c_list[l]))
        att_ws.append(att_w)

    # convert to numpy array with the shape (B, Lmax, Tmax)
    if isinstance(self.att, AttLoc2D):
        # att_ws => list of previous concate attentions
        att_ws = torch.stack([aw[:, -1] for aw in att_ws], dim=1).data.cpu().numpy()
    elif isinstance(self.att, (AttCov, AttCovLoc)):
        # att_ws => list of list of previous attentions
        att_ws = torch.stack([aw[-1] for aw in att_ws], dim=1).data.cpu().numpy()
    elif isinstance(self.att, AttLocRec):
        # att_ws => list of tuple of attention and hidden states
        att_ws = torch.stack([aw[0] for aw in att_ws], dim=1).data.cpu().numpy()
    elif isinstance(self.att, (AttMultiHeadDot, AttMultiHeadAdd, AttMultiHeadLoc, AttMultiHeadMultiResLoc)):
        # att_ws => list of list of each head attetion
        n_heads = len(att_ws[0])
        att_ws_sorted_by_head = []
        for h in six.moves.range(n_heads):
            att_ws_head = torch.stack([aw[h] for aw in att_ws], dim=1)
            att_ws_sorted_by_head += [att_ws_head]
        att_ws = torch.stack(att_ws_sorted_by_head, dim=1).data.cpu().numpy()
    else:
        # att_ws => list of attetions
        att_ws = torch.stack(att_ws, dim=1).data.cpu().numpy()
    return att_ws
