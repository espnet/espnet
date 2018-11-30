#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import logging
import math
import sys

from argparse import Namespace

import numpy as np
import random
import six

import chainer
import chainer.functions as F
import chainer.links as L

from chainer import reporter

from espnet.nets.chain.attentions import att_for_args
from espnet.nets.chain.ctc import ctc_for_args
import espnet.nets.chain.deterministic_embed_id as DL
from espnet.nets.chain.encoders import BLSTM
from espnet.nets.chain.encoders import BLSTMP
from espnet.nets.chain.encoders import VGG2L
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.e2e_asr_common import label_smoothing_dist

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5


# TODO(watanabe) merge Loss and E2E: there is no need to make these separately
class Loss(chainer.Chain):
    def __init__(self, predictor, mtlalpha):
        super(Loss, self).__init__()
        self.mtlalpha = mtlalpha
        self.loss = None
        self.accuracy = None

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, xs, ilens, ys):
        """Loss forward

        :param xs:
        :param ilens:
        :param ys:
        :return:
        """
        self.loss = None
        loss_ctc, loss_att, acc = self.predictor(xs, ilens, ys)
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
        elif alpha == 1:
            self.loss = loss_ctc
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att

        if self.loss.data < CTC_LOSS_THRESHOLD and not math.isnan(self.loss.data):
            reporter.report({'loss_ctc': loss_ctc}, self)
            reporter.report({'loss_att': loss_att}, self)
            reporter.report({'acc': acc}, self)

            logging.info('mtl loss:' + str(self.loss.data))
            reporter.report({'loss': self.loss}, self)
        else:
            logging.warning('loss (=%f) is not correct', self.loss.data)

        return self.loss


class E2E(chainer.Chain):
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

        with self.init_scope():
            # encoder
            self.enc = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs,
                               self.subsample, args.dropout_rate)
            # ctc
            self.ctc = ctc_for_args(args, odim)
            # attention
            self.att = att_for_args(args)
            # decoder
            self.dec = Decoder(args.eprojs, odim, args.dlayers, args.dunits,
                               self.sos, self.eos, self.att, self.verbose, self.char_list,
                               labeldist, args.lsm_weight, args.sampling_probability)

    def __call__(self, xs, ilens, ys):
        """E2E forward

        :param xs:
        :param ilens:
        :param ys:
        :return:
        """
        # 1. encoder
        hs, ilens = self.enc(xs, ilens)

        # 3. CTC loss
        if self.mtlalpha == 0:
            loss_ctc = None
        else:
            loss_ctc = self.ctc(hs, ys)

        # 4. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
        else:
            loss_att, acc = self.dec(hs, ys)

        return loss_ctc, loss_att, acc

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        """E2E greedy/beam search

        :param x:
        :param recog_args:
        :param char_list:
        :param rnnlm
        :return:
        """
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = self.xp.array(x.shape[0], dtype=np.int32)
        h = chainer.Variable(self.xp.array(x, dtype=np.float32))

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            # 1. encoder
            # make a utt list (1) to use the same interface for encoder
            h, _ = self.enc([h], [ilen])

            # calculate log P(z_t|X) for CTC scores
            if recog_args.ctc_weight > 0.0:
                lpz = self.ctc.log_softmax(h).data[0]
            else:
                lpz = None

            # 2. decoder
            # decode the first utterance
            y = self.dec.recognize_beam(h[0], lpz, recog_args, char_list, rnnlm)

            return y

    def calculate_all_attentions(self, xs, ilens, ys):
        """E2E attention calculation

        :param xs:
        :param list xs: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param np.ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: attention weights (B, Lmax, Tmax)
        :rtype: float np.ndarray
        """
        hs, ilens = self.enc(xs, ilens)
        att_ws = self.dec.calculate_all_attentions(hs, ys)

        return att_ws


# ------------- Decoder Network ----------------------------------------------------------------------------------------
class Decoder(chainer.Chain):
    def __init__(self, eprojs, odim, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.embed = DL.EmbedID(odim, dunits)
            self.lstm0 = L.StatelessLSTM(dunits + eprojs, dunits)
            for l in six.moves.range(1, dlayers):
                setattr(self, 'lstm%d' % l, L.StatelessLSTM(dunits, dunits))
            self.output = L.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dlayers = dlayers
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

    def __call__(self, hs, ys):
        """Decoder forward

        :param Variable hs:
        :param Variable ys:
        :return:
        """
        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = self.xp.array([self.eos], 'i')
        sos = self.xp.array([self.sos], 'i')
        ys_in = [F.concat([sos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = F.pad_sequence(ys_in, padding=self.eos)
        pad_ys_out = F.pad_sequence(ys_out, padding=-1)

        # get dim, length info
        batch = pad_ys_out.shape[0]
        olength = pad_ys_out.shape[1]
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(self.xp.array([h.shape[0] for h in hs])))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str(self.xp.array([y.shape[0] for y in ys_out])))

        # initialization
        c_list = [None]  # list of cell state of each layer
        z_list = [None]  # list of hidden state of each layer
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(None)
            z_list.append(None)
        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim
        eys = F.separate(eys, axis=1)

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs, z_list[0], att_w)
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(' scheduled sampling ')
                z_out = self.output(z_all[-1])
                z_out = F.argmax(F.log_softmax(z_out), axis=1)
                z_out = self.embed(z_out)
                ey = F.hstack((z_out, att_c))  # utt x (zdim + hdim)
            else:
                ey = F.hstack((eys[i], att_c))  # utt x (zdim + hdim)
            c_list[0], z_list[0] = self.lstm0(c_list[0], z_list[0], ey)
            for l in six.moves.range(1, self.dlayers):
                c_list[l], z_list[l] = self['lstm%d' % l](c_list[l], z_list[l], z_list[l - 1])
            z_all.append(z_list[-1])

        z_all = F.reshape(F.stack(z_all, axis=1),
                          (batch * olength, self.dunits))
        # compute loss
        y_all = self.output(z_all)
        self.loss = F.softmax_cross_entropy(y_all, F.flatten(pad_ys_out))
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = F.accuracy(y_all, F.flatten(pad_ys_out), ignore_label=-1)
        logging.info('att loss:' + str(self.loss.data))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            y_hat = F.reshape(y_all, (batch, olength, -1))
            y_true = pad_ys_out
            for (i, y_hat_), y_true_ in zip(enumerate(y_hat.data), y_true.data):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = self.xp.argmax(y_hat_[y_true_ != -1], axis=1)
                idx_true = y_true_[y_true_ != -1]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat).replace('<space>', ' ')
                seq_true = "".join(seq_true).replace('<space>', ' ')
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = chainer.Variable(self.xp.asarray(self.labeldist))
            loss_reg = - F.sum(F.scale(F.log_softmax(y_all), self.vlabeldist, axis=1)) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None):
        """beam search implementation

        :param h:
        :param lpz:
        :param recog_args:
        :param char_list:
        :param rnnlm:
        :return:
        """
        logging.info('input lengths: ' + str(h.shape[0]))
        # initialization
        c_list = [None]  # list of cell state of each layer
        z_list = [None]  # list of hidden state of each layer
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(None)
            z_list.append(None)
        a = None
        self.att.reset()  # reset pre-computation of h

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprate sos
        y = self.xp.full(1, self.sos, 'i')
        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.shape[0]))
        minlen = int(recog_args.minlenratio * h.shape[0])
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a, 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz, 0, self.eos, self.xp)
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
                ey = self.embed(hyp['yseq'][i])  # utt list (1) x zdim
                att_c, att_w = self.att([h], hyp['z_prev'][0], hyp['a_prev'])
                ey = F.hstack((ey, att_c))  # utt(1) x (zdim + hdim)
                c_list[0], z_list[0] = self.lstm0(hyp['c_prev'][0], hyp['z_prev'][0], ey)
                for l in six.moves.range(1, self.dlayers):
                    c_list[l], z_list[l] = self['lstm%d' % l](
                        hyp['c_prev'][l], hyp['z_prev'][l], z_list[l - 1])

                # get nbest local scores and their ids
                local_att_scores = F.log_softmax(self.output(z_list[-1])).data
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], hyp['yseq'][i])
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][:ctc_beam]
                    ctc_scores, ctc_states = ctc_prefix_score(hyp['yseq'], local_best_ids, hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids] \
                        + ctc_weight * (ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids]
                    joint_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][:beam]
                    local_best_scores = local_scores[:, joint_best_ids]
                    local_best_ids = local_best_ids[joint_best_ids]
                else:
                    local_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][:beam]
                    local_best_scores = local_scores[:, local_best_ids]

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # do not copy {z,c}_list directly
                    new_hyp['z_prev'] = z_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_w
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = self.xp.full(
                        1, local_best_ids[j], 'i')
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypotheses: ' + str(len(hyps)))
            logging.debug('best hypo: ' + ''.join([char_list[int(x)]
                                                   for x in hyps[0]['yseq'][1:]]).replace('<space>', ' '))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last position in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.xp.full(1, self.eos, 'i'))

            # add ended hypotheses to a final list, and removed them from current hypotheses
            # (this will be a problem, number of hyps < beam)
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
                logging.debug('remaining hypotheses: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug('hypo: ' + ''.join([char_list[int(x)]
                                                  for x in hyp['yseq'][1:]]).replace('<space>', ' '))

            logging.debug('number of ended hypotheses: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheses
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy because Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize_beam(h, lpz, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        return nbest_hyps

    def calculate_all_attentions(self, hs, ys):
        """Calculate all of attentions

        :return: list of attentions
        """
        # prepare input and output word sequences with sos/eos IDs
        eos = self.xp.array([self.eos], 'i')
        sos = self.xp.array([self.sos], 'i')
        ys_in = [F.concat([sos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = F.pad_sequence(ys_in, padding=self.eos)
        pad_ys_out = F.pad_sequence(ys_out, padding=-1)

        # get length info
        olength = pad_ys_out.shape[1]

        # initialization
        c_list = [None]  # list of cell state of each layer
        z_list = [None]  # list of hidden state of each layer
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(None)
            z_list.append(None)
        att_w = None
        att_ws = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim
        eys = F.separate(eys, axis=1)

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs, z_list[0], att_w)
            ey = F.hstack((eys[i], att_c))  # utt x (zdim + hdim)
            c_list[0], z_list[0] = self.lstm0(c_list[0], z_list[0], ey)
            for l in six.moves.range(1, self.dlayers):
                c_list[l], z_list[l] = self['lstm%d' % l](c_list[l], z_list[l], z_list[l - 1])
            att_ws.append(att_w)  # for debugging

        att_ws = F.stack(att_ws, axis=1)
        att_ws.to_cpu()

        return att_ws.data


# ------------- Encoder Network ----------------------------------------------------------------------------------------
class Encoder(chainer.Chain):
    """ENCODER NETWORK CLASS

    This is the example of docstring.

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int eprojs: number of projection units of encoder network
    :param np.ndarray subsample: subsampling number e.g. 1_2_2_2_1
    :param float dropout: dropout rate
    :return:

    """

    def __init__(self, etype, idim, elayers, eunits, eprojs, subsample, dropout, in_channel=1):
        super(Encoder, self).__init__()
        with self.init_scope():
            if etype == 'blstm':
                self.enc1 = BLSTM(idim, elayers, eunits, eprojs, dropout)
                logging.info('BLSTM without projection for encoder')
            elif etype == 'blstmp':
                self.enc1 = BLSTMP(idim, elayers, eunits,
                                   eprojs, subsample, dropout)
                logging.info('BLSTM with every-layer projection for encoder')
            elif etype == 'vggblstmp':
                self.enc1 = VGG2L(in_channel)
                self.enc2 = BLSTMP(get_vgg2l_odim(
                    idim, in_channel=in_channel), elayers, eunits, eprojs, subsample, dropout)
                logging.info('Use CNN-VGG + BLSTMP for encoder')
            elif etype == 'vggblstm':
                self.enc1 = VGG2L(in_channel)
                self.enc2 = BLSTM(get_vgg2l_odim(
                    idim, in_channel=in_channel), elayers, eunits, eprojs, dropout)
                logging.info('Use CNN-VGG + BLSTM for encoder')
            else:
                logging.error(
                    "Error: need to specify an appropriate encoder architecture")
                sys.exit()

        self.etype = etype

    def __call__(self, xs, ilens):
        """Encoder forward

        :param xs:
        :param ilens:
        :return:
        """
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
                "Error: need to specify an appropriate encoder architecture")
            sys.exit()

        return xs, ilens
