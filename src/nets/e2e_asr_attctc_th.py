#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging
import math
import sys

import chainer
import numpy as np
import six
import torch
import torch.nn.functional as F
import warpctc_pytorch as warp_ctc

from chainer import reporter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from ctc_prefix_score import CTCPrefixScore
from ctc_prefix_score import CTCPrefixScoreTH
from e2e_asr_common import end_detect
from e2e_asr_common import label_smoothing_dist


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


# get output dim for latter BLSTM
def _get_vgg2l_odim(idim, in_channel=3, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels


# get output dim for latter BLSTM
def _get_max_pooled_size(idim, out_channel=128, n_layers=2, ksize=2, stride=2):
    for _ in range(n_layers):
        idim = math.floor((idim - (ksize - 1) - 1) / stride)
    return idim  # numer of channels


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
    def report(self, loss_ctc, loss_att, acc, cer, wer, mtl_loss):
        reporter.report({'loss_ctc': loss_ctc}, self)
        reporter.report({'loss_att': loss_att}, self)
        reporter.report({'acc': acc}, self)
        reporter.report({'cer': cer}, self)
        reporter.report({'wer': wer}, self)
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
        loss_ctc, loss_att, acc, cer, wer = self.predictor(x)
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
            self.reporter.report(loss_ctc_data, loss_att_data, acc, cer, wer, loss_data)
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

    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def set_forget_bias_to_one(bias):
    n = bias.size(0)
    start, end = n // 4, n // 2
    bias.data[start:end].fill_(1.)


class E2E(torch.nn.Module):
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
        # additional forget-bias init in encoder ?
        # for m in self.modules():
        #     if isinstance(m, torch.nn.LSTM):
        #         for name, p in m.named_parameters():
        #             if "bias_ih" in name:
        #                 set_forget_bias_to_one(p)

        # options for beam search (trainig stage)
        if 'beam_size' in vars(args):
            recog_args = {'beam_size':args.beam_size, 'penalty':args.penalty,
                          'ctc_weight':args.ctc_weight, 'maxlenratio':args.maxlenratio,
                          'minlenratio':args.minlenratio, 'lm_weight':args.lm_weight,
                          'nbest':args.nbest}
            self.rnnlm = None
            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False

        self.logzero = -10000000000.0

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

        # 5. compute cer/wer
        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
            oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz = self.ctc.log_softmax(hpad).data
            else:
                lpz = None

            wers, cers = [], []
            nbest_hyps = self.dec.recognize_beam(hpad, hlens, lpz,
                                                 self.recog_args, self.char_list,
                                                 self.rnnlm)
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat]
                seq_true = [self.char_list[int(idx)] for idx in y_true]
                seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
                seq_hat_text = seq_hat_text.replace('<blank>', '').replace('<unk>', '')
                seq_true_text = "".join(seq_true).replace('<space>', ' ')

                # wer
                hyp_words = seq_hat_text.split()
                ref_words = seq_true_text.split()
                wers.append(editdistance.eval(hyp_words, ref_words) / len(ref_words))
                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                cers.append(editdistance.eval(hyp_chars, ref_chars) / len(ref_chars))

                if False:
                    logging.info("groundtruth: " + seq_true_text)
                    logging.info("prediction : " + seq_hat_text)
            wer = 0.0 if not self.report_wer else sum(wers) / len(wers)
            cer = 0.0 if not self.report_cer else sum(cers) / len(cers)

        return loss_ctc, loss_att, acc, cer, wer

    def recognize(self, xs, recog_args, char_list, rnnlm=None):
        '''E2E beam search

        :param x:
        :param recog_args:
        :param char_list:
        :return:
        '''
        prev = self.training
        self.eval()
        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        if torch_is_old:
            hs = [to_cuda(self, Variable(torch.from_numpy(np.array(xx, dtype=np.float32)),
                                         volatile=True))
                  for xx in xs]
        else:
            hs = [to_cuda(self, torch.from_numpy(np.array(xx, dtype=np.float32)))
                  for xx in xs]

        # 1. encoder
        xpad = pad_list(hs, 0.0)
        hpad, hlens = self.enc(xpad, ilens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(h).data
        else:
            lpz = None

        # 2. decoder
        y = self.dec.recognize_beam(hpad, hlens, lpz, recog_args, char_list, rnnlm)

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


# ------------- CTC Network --------------------------------------------------------------------------------------------
class _ChainerLikeCTC(warp_ctc._CTC):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens):
        is_cuda = True if acts.is_cuda else False
        acts = acts.contiguous()
        loss_func = warp_ctc.gpu_ctc if is_cuda else warp_ctc.cpu_ctc
        grads = torch.zeros(acts.size()).type_as(acts)
        minibatch_size = acts.size(1)
        costs = torch.zeros(minibatch_size).cpu()
        loss_func(acts,
                  grads,
                  labels,
                  label_lens,
                  act_lens,
                  minibatch_size,
                  costs)
        # modified only here from original
        costs = torch.FloatTensor([costs.sum()]) / acts.size(1)
        ctx.grads = Variable(grads)
        ctx.grads /= ctx.grads.size(1)

        return costs


def chainer_like_ctc_loss(acts, labels, act_lens, label_lens):
    """Chainer like CTC Loss

    acts: Tensor of (seqLength x batch x outputDim) containing output from network
    labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
    act_lens: Tensor of size (batch) containing size of each output sequence from the network
    act_lens: Tensor of (batch) containing label length of each example
    """
    assert len(labels.size()) == 1  # labels must be 1 dimensional
    from torch.nn.modules.loss import _assert_no_grad
    _assert_no_grad(labels)
    _assert_no_grad(act_lens)
    _assert_no_grad(label_lens)
    return _ChainerLikeCTC.apply(acts, labels, act_lens, label_lens)


class CTC(torch.nn.Module):
    def __init__(self, odim, eprojs, dropout_rate):
        super(CTC, self).__init__()
        self.dropout_rate = dropout_rate
        self.loss = None
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.loss_fn = chainer_like_ctc_loss  # CTCLoss()

    def forward(self, hpad, ilens, ys):
        '''CTC forward

        :param hs:
        :param ys:
        :return:
        '''
        self.loss = None
        ilens = Variable(torch.from_numpy(np.fromiter(ilens, dtype=np.int32)))
        olens = Variable(torch.from_numpy(np.fromiter(
            (x.size(0) for x in ys), dtype=np.int32)))

        # zero padding for hs
        y_hat = linear_tensor(
            self.ctc_lo, F.dropout(hpad, p=self.dropout_rate))

        # zero padding for ys
        y_true = torch.cat(ys).cpu().int()  # batch x olen

        # get length info
        logging.info(self.__class__.__name__ + ' input lengths:  ' + ''.join(str(ilens).split('\n')))
        logging.info(self.__class__.__name__ + ' output lengths: ' + ''.join(str(olens).split('\n')))

        # get ctc loss
        # expected shape of seqLength x batchSize x alphabet_size
        y_hat = y_hat.transpose(0, 1)
        self.loss = to_cuda(self, self.loss_fn(y_hat, y_true, ilens, olens))
        logging.info('ctc loss:' + str(self.loss.data[0]))

        return self.loss

    def log_softmax(self, hpad):
        '''log_softmax of frame activations

        :param hs:
        :return:
        '''
        return F.log_softmax(linear_tensor(self.ctc_lo, hpad), dim=2)


def mask_by_length(xs, length, fill=0):
    assert xs.size(0) == len(length)
    ret = Variable(xs.data.new(*xs.size()).fill_(fill))
    for i, l in enumerate(length):
        ret[i, :l] = xs[i, :l]
    return ret


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

        :param Variable enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param Variable dec_z: dummy (does not use)
        :param Variable att_prev: dummy (does not use)
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: Variable
        :return: previous attentioin weights
        :rtype: Variable
        '''
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [Variable(enc_hs_pad.data.new(
                l).zero_() + (1.0 / l)) for l in enc_hs_len]
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

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttDot forward

        :param Variable enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param Variable dec_z: dummy (does not use)
        :param Variable att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: Variable
        :return: previous attentioin weight (B x T_max)
        :rtype: Variable
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = torch.tanh(
                linear_tensor(self.mlp_enc, self.enc_h))

        if dec_z is None:
            dec_z = Variable(enc_hs_pad.data.new(batch, self.dunits).zero_())
        else:
            dec_z = dec_z.view(batch, self.dunits)

        e = torch.sum(self.pre_compute_enc_h * torch.tanh(self.mlp_dec(dec_z)).view(batch, 1, self.att_dim),
                      dim=2)  # utt x frame
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

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttLoc forward

        :param Variable enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param Variable dec_z: docoder hidden state (B x D_dec)
        :param Variable att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: Variable
        :return: previous attentioin weights (B x T_max)
        :rtype: Variable
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = linear_tensor(self.mlp_enc, self.enc_h)

        if dec_z is None:
            dec_z = Variable(enc_hs_pad.data.new(batch, self.dunits).zero_())
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        # NOTE consider zero padding when compute w.
        e = linear_tensor(self.gvec, torch.tanh(
            self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)
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
        self.aconv_chans = aconv_chans

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttLoc forward

        :param Variable enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param Variable dec_z: docoder hidden state (B x D_dec)
        :param Variable att_prev: previous attetion weight (B x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: Variable
        :return: previous attentioin weights (B x T_max)
        :rtype: Variable
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = linear_tensor(self.mlp_enc, self.enc_h)

        if dec_z is None:
            dec_z = Variable(enc_hs_pad.data.new(batch, self.dunits).zero_())
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [Variable(enc_hs_pad.data.new(
                l).zero_() + (1.0 / l)) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev = pad_list(att_prev, 0)

        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = linear_tensor(self.mlp_att, att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        # NOTE consider zero padding when compute w.
        e = linear_tensor(self.gvec, torch.tanh(
            att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)
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

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_list, scaling=2.0):
        '''AttCov forward

        :param Variable enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param Variable dec_z: docoder hidden state (B x D_dec)
        :param list att_prev_list: list of previous attetion weight
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: Variable
        :return: list of previous attentioin weights
        :rtype: list
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = linear_tensor(self.mlp_enc, self.enc_h)

        if dec_z is None:
            dec_z = Variable(enc_hs_pad.data.new(batch, self.dunits).zero_())
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev_list is None:
            att_prev = [Variable(enc_hs_pad.data.new(
                l).zero_() + (1.0 / l)) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev_list = [pad_list(att_prev, 0)]

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = sum(att_prev_list)
        # cov_vec: B x T => B x T x 1 => B x T x att_dim
        cov_vec = linear_tensor(self.wvec, cov_vec.unsqueeze(-1))

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        # NOTE consider zero padding when compute w.
        e = linear_tensor(self.gvec, torch.tanh(
            cov_vec + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

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

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttLoc2D forward

        :param Variable enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param Variable dec_z: docoder hidden state (B x D_dec)
        :param Variable att_prev: previous attetion weight (B x att_win x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: Variable
        :return: previous attentioin weights (B x att_win x T_max)
        :rtype: Variable
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = linear_tensor(self.mlp_enc, self.enc_h)

        if dec_z is None:
            dec_z = Variable(enc_hs_pad.data.new(batch, self.dunits).zero_())
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # B * [Li x att_win]
            att_prev = [Variable(
                enc_hs_pad.data.new(l, self.att_win).zero_() + 1.0 / l) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev = pad_list(att_prev, 0).transpose(1, 2)

        # att_prev: B x att_win x Tmax -> B x 1 x att_win x Tmax -> B x C x 1 x Tmax
        att_conv = self.loc_conv(att_prev.unsqueeze(1))
        # att_conv: B x C x 1 x Tmax -> B x Tmax x C
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = linear_tensor(self.mlp_att, att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        # NOTE consider zero padding when compute w.
        e = linear_tensor(self.gvec, torch.tanh(
            att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

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

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_states, scaling=2.0):
        '''AttLocRec forward

        :param Variable enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param Variable dec_z: docoder hidden state (B x D_dec)
        :param tuple att_prev_states: previous attetion weight and lstm states
                                      ((B, T_max), ((B, att_dim), (B, att_dim)))
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: Variable
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
            self.pre_compute_enc_h = linear_tensor(self.mlp_enc, self.enc_h)

        if dec_z is None:
            dec_z = Variable(enc_hs_pad.data.new(batch, self.dunits).zero_())
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev_states is None:
            # initialize attention weight with uniform dist.
            att_prev = [Variable(
                enc_hs_pad.data.new(l).fill_(1.0 / l)) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev = pad_list(att_prev, 0)

            # initialize lstm states
            att_h = Variable(enc_hs_pad.data.new(batch, self.att_dim).zero_())
            att_c = Variable(enc_hs_pad.data.new(batch, self.att_dim).zero_())
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
        # NOTE consider zero padding when compute w.
        e = linear_tensor(self.gvec, torch.tanh(
            att_h.unsqueeze(1) + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

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

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_list, scaling=2.0):
        '''AttCovLoc forward

        :param Variable enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param Variable dec_z: docoder hidden state (B x D_dec)
        :param list att_prev_list: list of previous attetion weight
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: Variable
        :return: list of previous attentioin weights
        :rtype: list
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = linear_tensor(self.mlp_enc, self.enc_h)

        if dec_z is None:
            dec_z = Variable(enc_hs_pad.data.new(batch, self.dunits).zero_())
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev_list is None:
            att_prev = [Variable(enc_hs_pad.data.new(
                l).zero_() + (1.0 / l)) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev_list = [pad_list(att_prev, 0)]

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = sum(att_prev_list)

        # cov_vec: B x T -> B x 1 x 1 x T -> B x C x 1 x T
        att_conv = self.loc_conv(cov_vec.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = linear_tensor(self.mlp_att, att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        # NOTE consider zero padding when compute w.
        e = linear_tensor(self.gvec, torch.tanh(
            att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

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

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''AttMultiHeadDot forward

        :param Variable enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param Variable dec_z: decoder hidden state (B x D_dec)
        :param Variable att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B x D_enc)
        :rtype: Variable
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
                torch.tanh(linear_tensor(self.mlp_k[h], self.enc_h)) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                linear_tensor(self.mlp_v[h], self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = Variable(enc_hs_pad.data.new(batch, self.dunits).zero_())
        else:
            dec_z = dec_z.view(batch, self.dunits)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            e = torch.sum(self.pre_compute_k[h] * torch.tanh(self.mlp_q[h](dec_z)).view(
                batch, 1, self.att_dim_k), dim=2)  # utt x frame
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

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''AttMultiHeadAdd forward

        :param Variable enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param Variable dec_z: decoder hidden state (B x D_dec)
        :param Variable att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: Variable
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
                linear_tensor(self.mlp_k[h], self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                linear_tensor(self.mlp_v[h], self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = Variable(enc_hs_pad.data.new(batch, self.dunits).zero_())
        else:
            dec_z = dec_z.view(batch, self.dunits)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            e = linear_tensor(
                self.gvec[h],
                torch.tanh(
                    self.pre_compute_k[h] + self.mlp_q[h](dec_z).view(batch, 1, self.att_dim_k))).squeeze(2)
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

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttMultiHeadLoc forward

        :param Variable enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param Variable dec_z: decoder hidden state (B x D_dec)
        :param Variable att_prev: list of previous attentioin weight (B x T_max) * aheads
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B x D_enc)
        :rtype: Variable
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
                linear_tensor(self.mlp_k[h], self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                linear_tensor(self.mlp_v[h], self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = Variable(enc_hs_pad.data.new(batch, self.dunits).zero_())
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            att_prev = []
            for h in six.moves.range(self.aheads):
                att_prev += [[Variable(enc_hs_pad.data.new(
                    l).zero_() + (1.0 / l)) for l in enc_hs_len]]
                # if no bias, 0 0-pad goes 0
                att_prev[h] = pad_list(att_prev[h], 0)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch, 1, 1, self.h_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = linear_tensor(self.mlp_att[h], att_conv)

            e = linear_tensor(
                self.gvec[h],
                torch.tanh(
                    self.pre_compute_k[h] + att_conv + self.mlp_q[h](dec_z).view(
                        batch, 1, self.att_dim_k))).squeeze(2)
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

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''AttMultiHeadMultiResLoc forward

        :param Variable enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param Variable dec_z: decoder hidden state (B x D_dec)
        :param Variable att_prev: list of previous attentioin weight (B x T_max) * aheads
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B x D_enc)
        :rtype: Variable
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
                linear_tensor(self.mlp_k[h], self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                linear_tensor(self.mlp_v[h], self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = Variable(enc_hs_pad.data.new(batch, self.dunits).zero_())
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            att_prev = []
            for h in six.moves.range(self.aheads):
                att_prev += [[Variable(enc_hs_pad.data.new(
                    l).zero_() + (1.0 / l)) for l in enc_hs_len]]
                # if no bias, 0 0-pad goes 0
                att_prev[h] = pad_list(att_prev[h], 0)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch, 1, 1, self.h_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = linear_tensor(self.mlp_att[h], att_conv)

            e = linear_tensor(
                self.gvec[h],
                torch.tanh(
                    self.pre_compute_k[h] + att_conv + self.mlp_q[h](dec_z).view(
                        batch, 1, self.att_dim_k))).squeeze(2)
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


def th_accuracy(y_all, pad_target, ignore_label):
    pad_pred = y_all.data.view(pad_target.size(
        0), pad_target.size(1), y_all.size(1)).max(2)[1]
    mask = pad_target.data != ignore_label
    numerator = torch.sum(pad_pred.masked_select(
        mask) == pad_target.data.masked_select(mask))
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


def get_last_yseq(exp_yseq):
    last = []
    for y_seq in exp_yseq:
        last.append(y_seq[-1])
    return last


def append_ids(yseq, ids):
    if isinstance(ids, list):
        for i, j in enumerate(ids):
            yseq[i].append(j)
    else:
        for i in range(len(yseq)):
            yseq[i].append(ids)
    return yseq


def expand_yseq(yseqs, next_ids):
    new_yseq = []
    for yseq in yseqs:
        for next_id in next_ids:
            new_yseq.append(yseq[:])
            new_yseq[-1].append(next_id)
    return new_yseq


def index_select_list(yseq, lst):
    new_yseq = []
    for l in lst:
        new_yseq.append(yseq[l][:])
    return new_yseq


def index_select_lm_state(rnnlm_state, dim, vidx):
    state = dict([(k, torch.index_select(v, dim, vidx))
                  for k,v in rnnlm_state.items()])
    return state


# ------------- Decoder Network ----------------------------------------------------------------------------------------
class Decoder(torch.nn.Module):
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
        self.odim = odim
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight

        self.logzero = -10000000000.0

    def zero_state(self, hpad):
        return Variable(hpad.data.new(hpad.size(0), self.dunits).zero_())

    def forward(self, hpad, hlen, ys):
        '''Decoder forward

        :param hs:
        :param ys:
        :return:
        '''
        hpad = mask_by_length(hpad, hlen, 0)
        hlen = list(map(int, hlen))

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

        # get dim, length info
        batch = pad_ys_out.size(0)
        olength = pad_ys_out.size(1)
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(hlen))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out]))

        # initialization
        c_list = [self.zero_state(hpad)]
        z_list = [self.zero_state(hpad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hpad))
            z_list.append(self.zero_state(hpad))
        att_w = None
        z_all = []
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
            z_all.append(z_list[-1])

        z_all = torch.stack(z_all, dim=1).view(batch * olength, self.dunits)
        # compute loss
        y_all = self.output(z_all)
        self.loss = F.cross_entropy(y_all, pad_ys_out.view(-1),
                                    ignore_index=self.ignore_id,
                                    size_average=True)
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = th_accuracy(y_all, pad_ys_out, ignore_label=self.ignore_id)
        logging.info('att loss:' + ''.join(str(self.loss.data).split('\n')))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            y_hat = y_all.view(batch, olength, -1)
            y_true = pad_ys_out
            for (i, y_hat_), y_true_ in zip(enumerate(y_hat.data.cpu().numpy()),
                                            y_true.data.cpu().numpy()):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat_[y_true_ != self.ignore_id], axis=1)
                idx_true = y_true_[y_true_ != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_cuda(self, Variable(torch.from_numpy(self.labeldist)))
            loss_reg = - torch.sum((F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc

    def recognize_beam(self, h, hlens, lpz, recog_args, char_list, rnnlm=None):
        '''beam search implementation

        :param Variable h:
        :param Namespace recog_args:
        :param char_list:
        :return:
        '''
        logging.info('input lengths: ' + str(h.size(1)))
        h = mask_by_length(h, hlens, 0)

        # search params
        batch = len(hlens)
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        att_weight = 1.0 - ctc_weight

        n_bb = batch * beam
        n_bo = beam * self.odim
        n_bbo = n_bb * self.odim
        if torch_is_old:
            pad_b = to_cuda(self, Variable(torch.LongTensor([i * beam for i in six.moves.range(batch)]).view(-1, 1)))
            pad_bo = to_cuda(self, Variable(torch.LongTensor([i * n_bo for i in six.moves.range(batch)]).view(-1, 1)))
        else:
            pad_b = to_cuda(self, torch.LongTensor([i * beam for i in six.moves.range(batch)]).view(-1, 1))
            pad_bo = to_cuda(self, torch.LongTensor([i * n_bo for i in six.moves.range(batch)]).view(-1, 1))
        beam0 = torch.LongTensor([0 for _ in range(n_bb)])

        max_hlen = max(hlens)
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialization
        if torch_is_old:
            c_prev = [to_cuda(self, Variable(torch.zeros(n_bb, self.dunits), volatile=True))
                      for _ in range(self.dlayers)]
            z_prev = [to_cuda(self, Variable(torch.zeros(n_bb, self.dunits), volatile=True))
                      for _ in range(self.dlayers)]
            c_list = [to_cuda(self, Variable(torch.zeros(n_bb, self.dunits), volatile=True))
                      for _ in range(self.dlayers)]
            z_list = [to_cuda(self, Variable(torch.zeros(n_bb, self.dunits), volatile=True))
                      for _ in range(self.dlayers)]
            vscores = to_cuda(self, Variable(torch.zeros(batch, beam), volatile=True))
        else:
            c_prev = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
            z_prev = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
            c_list = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
            z_list = [to_cuda(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
            vscores = to_cuda(self, torch.zeros(batch, beam))
        a_prev = None
        rnnlm_prev = None

        self.att.reset()  # reset pre-computation of h

        # prepare search related variable
        yseq = [[self.sos] for _ in range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]

        ended_hyps = [[] for _ in range(batch)]
        if torch_is_old:
            exp_hlens = torch.LongTensor(hlens * beam).view(beam, batch).transpose(0, 1).contiguous()
        else:
            exp_hlens = hlens.repeat(beam).view(beam, batch).transpose(0, 1).contiguous()
        exp_hlens = exp_hlens.view(-1).tolist()
        exp_h = h.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
        exp_h = exp_h.view(n_bb, h.size()[1], h.size()[2])

        if lpz is not None:
            ctc_prefix_score = CTCPrefixScoreTH(lpz, 0, self.eos, beam, exp_hlens, lpz.is_cuda)
            ctc_states_prev = ctc_prefix_score.initial_state()
            ctc_scores_prev = to_cuda(self, torch.zeros(batch, n_bo))
            if torch_is_old:
                ctc_states_prev = Variable(ctc_states_prev)
                ctc_scores_prev = Variable(ctc_scores_prev, volatile=True))

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            if torch_is_old:
                vy = to_cuda(self, Variable(torch.LongTensor(get_last_yseq(yseq)), volatile=True))
            else:
                vy = to_cuda(self, torch.LongTensor(get_last_yseq(yseq)))
            ey = self.embed(vy)
            att_c, att_w = self.att(exp_h, exp_hlens, z_prev[0], a_prev)
            ey = torch.cat((ey, att_c), dim=1)

            # attention decoder
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](z_list[l - 1], (z_prev[l], c_prev[l]))
            local_scores = att_weight * F.log_softmax(self.output(z_list[-1]), dim=1)

            # rnnlm
            if rnnlm:
                rnnlm_state, z_rnnlm = rnnlm.predictor(rnnlm_prev, vy)
                local_lm_scores = F.log_softmax(z_rnnlm, dim=1)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores
            local_scores = local_scores.view(batch, n_bo)

            # ctc
            if lpz is not None:
                ctc_scores, ctc_states = ctc_prefix_score(yseq, ctc_states_prev.data)
                ctc_scores = ctc_scores.view(batch, n_bo)
                if torch_is_old:
                    ctc_scores = Variable(ctc_scores, volatile=True)
                    ctc_states = Variable(ctc_states, volatile=True)
                local_scores = local_scores + ctc_weight * (ctc_scores - ctc_scores_prev)

            local_best_scores = []
            local_best_odims = []
            local_padded_beam_ids = []
            local_padded_best_ids = []
            local_scores = local_scores.view(batch, beam, self.odim)

            if i == 0:
                local_scores = local_scores[:,0,:]
                local_best_scores, local_best_odims = torch.topk(local_scores, beam, 1)
                yseq = append_ids(yseq, local_best_odims.view(-1).data.cpu().tolist())
                vidx = to_cuda(self, torch.LongTensor(beam0))
                if torch_is_old:
                    vidx = Variable(vidx, volatile=True)
                local_padded_odim_ids = []
                local_padded_odim_ids = (torch.fmod(local_best_odims, self.odim) + pad_bo).view(-1).data.cpu().tolist()
                vscores = local_best_scores
            else:
                local_best_scores, local_best_odims = torch.topk(local_scores.view(batch, beam, self.odim),
                                                                 beam, 2)
                if torch_is_old:
                    local_scores = to_cuda(self, Variable(torch.FloatTensor(batch, beam, self.odim)))
                    local_scores[:,:,:] = self.logzero
                    _best_odims = local_best_odims.data
                    _best_score = local_best_scores.data
                else:
                    local_scures = torch.full((batch, beam, self.odim), self.logzero)
                    _best_odims = local_best_odims
                    _best_score = local_best_scores
                    
                # :should be modified
                for si in six.moves.range(batch):
                    for bj in six.moves.range(beam):
                        for bk in six.moves.range(beam):
                            local_scores[si,bj,_best_odims[si,bj,bk]] = \
                                        _best_score[si,bj,bk]

                vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim)
                vscores = (vscores + local_scores).view(batch, n_bo)

                accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1)
                local_padded_beam_ids, local_padded_odim_ids = [], []
                local_odim_ids = torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
                local_padded_odim_ids = (torch.fmod(accum_best_ids, self.odim) + pad_bo).view(-1).data.cpu().tolist()
                local_padded_beam_ids = (torch.div(accum_best_ids, self.odim) + pad_b).view(-1).data.cpu().tolist()

                yseq = index_select_list(yseq, local_padded_beam_ids)
                yseq = append_ids(yseq, local_odim_ids)
                vscores = accum_best_scores
                vidx = to_cuda(self, torch.LongTensor(local_padded_beam_ids))
                if torch_is_old:
                    vidx = Variable(vidx, volatile=True)

            a_prev = torch.index_select(att_w.view(n_bb, -1), 0, vidx)
            z_prev = [torch.index_select(z_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]
            c_prev = [torch.index_select(c_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]

            if rnnlm:
                rnnlm_prev = index_select_lm_state(rnnlm_state, 0, vidx)
            if lpz is not None:
                ctc_vidx = to_cuda(self, torch.LongTensor(local_padded_odim_ids))
                if torch_is_old:
                    ctc_vidx = Variable(ctc_vidx, volatile=True)
                ctc_scores_prev = torch.index_select(ctc_scores.view(-1), 0, ctc_vidx)
                ctc_scores_prev = ctc_scores_prev.view(-1, 1).repeat(1, self.odim).view(batch, n_bo)

                ctc_states = torch.transpose(ctc_states, 1, 3).contiguous()
                ctc_states = ctc_states.view(n_bbo, 2, -1)
                ctc_states_prev = torch.index_select(ctc_states, 0, ctc_vidx).view(n_bb, 2, -1)
                ctc_states_prev = torch.transpose(ctc_states_prev, 1, 2)

            # add ended hypothes to a final list, and removed them from current hypothes
            penalty_i = (i + 1) * penalty
            vscore_list = list(torch.split(vscores.view(-1), 1, dim=0))

            # pick ended hyps
            for y_id, y_hyp in enumerate(yseq):
                batch_idx = int(y_id / beam)
                if stop_search[batch_idx] or hlens[batch_idx] < i:
                    continue
                #if y_id % beam == 0:  # beam_idx
                #    n_eos = 0
                if hlens[batch_idx] - 1 == i:
                    y_hyp.append(self.eos)

                if len(y_hyp) > minlen and y_hyp[-1] == self.eos \
                   and ((i+2) == len(y_hyp) or hlens[batch_idx] - 1 == i):
                    _vscore = vscore_list[y_id] + penalty_i
                    _score = _vscore.data.cpu().numpy()
                    ended_hyps[batch_idx].append({'yseq':y_hyp[:], 'vscore':_vscore, 'score':_score})
                    yseq[y_id] = [self.sos, self.eos]
                    vscore_list[y_id] = _vscore * 0.0 + self.logzero
                    """
                    n_eos += 1
                    if n_eos == beam:
                    stop_search[batch_idx] = True  # no hyps
                    """
            vscores = torch.cat(vscore_list).view(batch, beam)
            # end detection
            stop_search = [stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                           for samp_i in six.moves.range(batch)]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            torch.cuda.empty_cache()

        dummy_hyps = [{'yseq':[self.sos, self.eos], 'score':np.array([-float('inf')])}]
        ended_hyps = [ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
                      for samp_i in six.moves.range(batch)]
        nbest_hyps = [sorted(ended_hyps[samp_i], key=lambda x: x['score'],
                             reverse=True)[:min(len(ended_hyps[samp_i]), recog_args.nbest)]
                      for samp_i in six.moves.range(batch)]

        return nbest_hyps

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


# ------------- Encoder Network ----------------------------------------------------------------------------------------
class Encoder(torch.nn.Module):
    '''ENCODER NETWORK CLASS

    This is the example of docstring.

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int epojs: number of projection units of encoder network
    :param str subsample: subsampling number e.g. 1_2_2_2_1
    :param float dropout: dropout rate
    :return:

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
            self.enc2 = BLSTMP(_get_vgg2l_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            logging.info('Use CNN-VGG + BLSTMP for encoder')
        elif etype == 'vggblstm':
            self.enc1 = VGG2L(in_channel)
            self.enc2 = BLSTM(_get_vgg2l_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG + BLSTM for encoder')
        else:
            logging.error(
                "Error: need to specify an appropriate encoder archtecture")
            sys.exit()

        self.etype = etype

    def forward(self, xs, ilens):
        '''Encoder forward

        :param xs:
        :param ilens:
        :return:
        '''
        if self.etype == 'blstm':
            xs, ilens = self.enc1(xs, ilens)
        elif self.etype == 'blstmp':
            xs, ilens = self.enc1(xs, ilens)
        elif self.etype == 'vggblstmp':
            xs, ilens = self.enc1(xs, ilens)
            xs, ilens = self.enc2(xs, ilens)
        elif self.etype == 'vggblstm':
            xs, ilens = self.enc1(xs, ilens)
            xs, ilens = self.enc2(xs, ilens)
        else:
            logging.error(
                "Error: need to specify an appropriate encoder archtecture")
            sys.exit()

        return xs, ilens


class BLSTMP(torch.nn.Module):
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

    def forward(self, xpad, ilens):
        '''BLSTMP forward

        :param xs:
        :param ilens:
        :return:
        '''
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        for layer in six.moves.range(self.elayers):
            xpack = pack_padded_sequence(xpad, ilens, batch_first=True)
            bilstm = getattr(self, 'bilstm' + str(layer))
            if torch_is_old:
                # pytorch 0.4.x does not support flatten_parameters() for multiple GPUs
                bilstm.flatten_parameters()
            ys, (hy, cy) = bilstm(xpack)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ypad, ilens = pad_packed_sequence(ys, batch_first=True)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ypad = ypad[:, ::sub]
                ilens = [int(i + 1) // sub for i in ilens]
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'bt' + str(layer)
                                )(ypad.contiguous().view(-1, ypad.size(2)))
            xpad = torch.tanh(projected.view(ypad.size(0), ypad.size(1), -1))
            del hy, cy

        return xpad, ilens  # x: utt list of frame x dim


class BLSTM(torch.nn.Module):
    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(BLSTM, self).__init__()
        self.nblstm = torch.nn.LSTM(idim, cdim, elayers, batch_first=True,
                                    dropout=dropout, bidirectional=True)
        self.l_last = torch.nn.Linear(cdim * 2, hdim)

    def forward(self, xpad, ilens):
        '''BLSTM forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        xpack = pack_padded_sequence(xpad, ilens, batch_first=True)
        ys, (hy, cy) = self.nblstm(xpack)
        del hy, cy
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ypad, ilens = pad_packed_sequence(ys, batch_first=True)
        # (sum _utt frame_utt) x dim
        projected = torch.tanh(self.l_last(
            ypad.contiguous().view(-1, ypad.size(2))))
        xpad = projected.view(ypad.size(0), ypad.size(1), -1)
        return xpad, ilens  # x: utt list of frame x dim


class VGG2L(torch.nn.Module):
    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.in_channel = in_channel

    def forward(self, xs, ilens):
        '''VGG2L forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        # xs = F.pad_sequence(xs)

        # x: utt x 1 (input channel num) x frame x dim
        xs = xs.view(xs.size(0), xs.size(1), self.in_channel,
                     xs.size(2) // self.in_channel).transpose(1, 2)

        # NOTE: max_pool1d ?
        xs = F.relu(self.conv1_1(xs))
        xs = F.relu(self.conv1_2(xs))
        xs = F.max_pool2d(xs, 2, stride=2, ceil_mode=True)

        xs = F.relu(self.conv2_1(xs))
        xs = F.relu(self.conv2_2(xs))
        xs = F.max_pool2d(xs, 2, stride=2, ceil_mode=True)
        # change ilens accordingly
        # ilens = [_get_max_pooled_size(i) for i in ilens]
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = xs.transpose(1, 2)
        xs = xs.contiguous().view(
            xs.size(0), xs.size(1), xs.size(2) * xs.size(3))
        xs = [xs[i, :ilens[i]] for i in range(len(ilens))]
        xs = pad_list(xs, 0.0)
        return xs, ilens
