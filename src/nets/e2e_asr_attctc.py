#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys
import logging
import six
import math

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

CTC_LOSS_THRESHOLD = 10000
MAX_DECODER_OUTPUT = 5


def _ilens_to_index(ilens):
    x = np.zeros(len(ilens), dtype=np.int32)
    for i in range(1, len(ilens)):
        x[i] = x[i - 1] + ilens[i - 1]
    return x[1:]


def _subsamplex(x, n):
    x = [F.get_item(xx, (slice(None, None, n), slice(None))) for xx in x]
    ilens = [xx.shape[0] for xx in x]
    return x, ilens


# get output dim for latter BLSTM
def _get_vgg2l_odim(idim, in_channel=3, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    idim = np.array(idim, dtype=np.int32)
    return idim * out_channel  # numer of channels


def linear_tensor(linear, x):
    '''
    Apply linear matrix operation only for the last dimension of a tensor

    :param Link linear: Linear link (M x N matrix)
    :param Variable x: Tensor (D_1 x D_2 x ... x M matrix)
    :return:
    :param Variable x: Tensor (D_1 x D_2 x ... x N matrix)
    '''
    dim = 1
    shapes = list(x.shape[:-1])
    for d in shapes:
        dim = dim * d
    y = linear(F.reshape(x, (dim, x.shape[-1])))
    shapes.append(y.shape[-1])

    return F.reshape(y, shapes)


def sequence_embed(embed, xs):
    '''

    :param embed:
    :param xs:
    :return:
    '''
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


# TODO merge MTLLoss and E2E: there is no need to make these separately
class MTLLoss(chainer.Chain):
    def __init__(self, predictor, mtlalpha):
        super(MTLLoss, self).__init__()
        self.mtlalpha = mtlalpha
        self.loss = None
        self.accuracy = None

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x):
        '''

        :param x:
        :return:
        '''
        self.loss = None
        loss_ctc, loss_att, acc = self.predictor(x)
        alpha = self.mtlalpha
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

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        subsample = np.ones(args.elayers + 1, dtype=np.int)  # +1 means input (+1) and layers outputs (args.elayer)
        if args.etype == 'blstmp':
            ss = args.subsample.split("_")
            for j in range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning('Subsampling is not performed for vggblstmp. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        with self.init_scope():
            # encoder
            self.enc = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs,
                               self.subsample, args.dropout_rate)
            # ctc
            self.ctc = CTC(odim, args.eprojs, args.dropout_rate)
            # attention
            if args.atype == 'dot':
                self.att = AttDot(args.eprojs, args.dunits, args.adim)
            elif args.atype == 'location':
                self.att = AttLoc(args.eprojs, args.dunits, args.adim, args.aconv_chans, args.aconv_filts)
            else:
                logging.error("Error: need to specify an appropriate attention archtecture")
                sys.exit()
            # decoder
            self.dec = Decoder(args.eprojs, odim, args.dlayers, args.dunits,
                               self.sos, self.eos, self.att, self.verbose, self.char_list)

    # x[i]: ('utt_id', {'ilen':'xxx',...}})
    def __call__(self, data):
        '''

        :param data:
        :return:
        '''
        # utt list of frame x dim
        xs = [i[1]['feat'] for i in data]
        # utt list of olen
        ys = [self.xp.array(map(float, i[1]['tokenid'].split()), dtype=np.int32) for i in data]
        ys = [chainer.Variable(y) for y in ys]

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = self.xp.array([xx.shape[0] for xx in xs], dtype=np.int32)
        hs = [chainer.Variable(self.xp.array(xx, dtype=np.float32)) for xx in xs]

        # 1. encoder
        hs, ilens = self.enc(hs, ilens)

        # 3. CTC loss
        loss_ctc = self.ctc(hs, ys)

        # 4. attention loss
        loss_att, acc, att_w = self.dec(hs, ys)

        # get alignment
        '''
        if self.verbose > 0 and self.outdir is not None:
            for i in six.moves.range(len(data)):
                utt = data[i][0]
                align_file = self.outdir + '/' + utt + '.ali'
                with open(align_file, "w") as f:
                    logging.info('writing an alignment file to' + align_file)
                    pickle.dump((utt, att_w[i]), f)
        '''

        return loss_ctc, loss_att, acc

    def recognize(self, x, recog_args, char_list):
        '''

        :param x:
        :param recog_args:
        :param char_list:
        :return:
        '''
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = self.xp.array(x.shape[0], dtype=np.int32)
        h = chainer.Variable(self.xp.array(x, dtype=np.float32))

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            # 1. encoder
            # make a utt list (1) to use the same interface for encoder
            h, _ = self.enc([h], [ilen])

            # 2. decoder
            # decode the first utterance
            if recog_args.beam_size == 1:
                y = self.dec.recognize(h[0], recog_args)
            else:
                y = self.dec.recognize_beam(h[0], recog_args, char_list)

            return y


# ------------- CTC Network --------------------------------------------------------------------------------------------
class CTC(chainer.Chain):
    def __init__(self, odim, eprojs, dropout_rate):
        super(CTC, self).__init__()
        self.dropout_rate = dropout_rate
        self.loss = None

        with self.init_scope():
            self.ctc_lo = L.Linear(eprojs, odim)

    def __call__(self, hs, ys):
        '''

        :param hs:
        :param ys:
        :return:
        '''
        self.loss = None
        ilens = [x.shape[0] for x in hs]
        olens = [x.shape[0] for x in ys]

        # zero padding for hs
        y_hat = linear_tensor(self.ctc_lo, F.dropout(F.pad_sequence(hs), ratio=self.dropout_rate))
        y_hat = F.separate(y_hat, axis=1)  # ilen list of batch x hdim

        # zero padding for ys
        y_true = F.pad_sequence(ys, padding=-1)  # batch x olen

        # get length info
        input_length = chainer.Variable(self.xp.array(ilens, dtype=np.int32))
        label_length = chainer.Variable(self.xp.array(olens, dtype=np.int32))
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(input_length.data))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str(label_length.data))

        # get ctc loss
        self.loss = F.connectionist_temporal_classification(y_hat, y_true, 0, input_length, label_length)
        logging.info('ctc loss:' + str(self.loss.data))

        return self.loss


# ------------- Attention Network --------------------------------------------------------------------------------------
# dot product based attention
class AttDot(chainer.Chain):
    def __init__(self, eprojs, dunits, att_dim):
        super(AttDot, self).__init__()
        with self.init_scope():
            self.mlp_enc = L.Linear(eprojs, att_dim)
            self.mlp_dec = L.Linear(dunits, att_dim)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def reset(self):
        '''

        :return:
        '''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def __call__(self, enc_hs, dec_z, scaling=2.0):
        '''

        :param enc_hs:
        :param dec_z:
        :param scaling:
        :return:
        '''
        batch = len(enc_hs)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = F.tanh(linear_tensor(self.mlp_enc, self.enc_h))

        if dec_z is None:
            dec_z = chainer.Variable(self.xp.zeros((batch, self.dunits), dtype=np.float32))
        else:
            dec_z = F.reshape(dec_z, (batch, self.dunits))

        # <phi (h_t), psi (s)> for all t
        e = F.sum(self.pre_compute_enc_h * F.tile(F.reshape(F.tanh(self.mlp_dec(dec_z)), (batch, 1, self.att_dim)),
                                                  (1, self.h_length, 1)), axis=2)  # utt x frame
        w = F.softmax(scaling * e)
        # weighted sum over flames
        # utt x hdim
        c = F.sum(self.enc_h * F.tile(F.reshape(w, (batch, self.h_length, 1)), (1, 1, self.eprojs)), axis=1)

        return c, w


# location based attention
class AttLoc(chainer.Chain):
    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttLoc, self).__init__()
        with self.init_scope():
            self.mlp_enc = L.Linear(eprojs, att_dim)
            self.mlp_dec = L.Linear(dunits, att_dim, nobias=True)
            self.mlp_att = L.Linear(aconv_chans, att_dim, nobias=True)
            self.loc_conv = L.Convolution2D(1, aconv_chans, ksize=(1, 2 * aconv_filts + 1), pad=(0, aconv_filts))
            self.gvec = L.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans

    def reset(self):
        '''

        :return:
        '''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def __call__(self, enc_hs, dec_z, att_prev, scaling=2.0):
        '''

        :param enc_hs:
        :param dec_z:
        :param att_prev:
        :param scaling:
        :return:
        '''
        batch = len(enc_hs)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = linear_tensor(self.mlp_enc, self.enc_h)

        if dec_z is None:
            dec_z = chainer.Variable(self.xp.zeros((batch, self.dunits), dtype=np.float32))
        else:
            dec_z = F.reshape(dec_z, (batch, self.dunits))

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [self.xp.ones(hh.shape[0], dtype=np.float32) * (1.0 / hh.shape[0]) for hh in enc_hs]
            att_prev = [chainer.Variable(att) for att in att_prev]
            att_prev = F.pad_sequence(att_prev)

        # TODO use <chainer variable>.reshpae(), instead of F.reshape()
        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(F.reshape(att_prev, (batch, 1, 1, self.h_length)))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = F.swapaxes(F.squeeze(att_conv, axis=2), 1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = linear_tensor(self.mlp_att, att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = F.tile(F.reshape(self.mlp_dec(dec_z), (batch, 1, self.att_dim)), (1, self.h_length, 1))

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        # TODO consider zero padding when compute w.
        e = F.squeeze(linear_tensor(self.gvec, F.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)), axis=2)
        w = F.softmax(scaling * e)

        # weighted sum over flames
        # utt x hdim
        c = F.sum(self.enc_h * F.tile(F.reshape(w, (batch, self.h_length, 1)), (1, 1, self.eprojs)), axis=1)

        return c, w


# ------------- Decoder Network ----------------------------------------------------------------------------------------
class Decoder(chainer.Chain):
    def __init__(self, eprojs, odim, dlayers, dunits, sos, eos, att, verbose=0, char_list=None):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(odim, dunits)
            # TODO use multiple layers with dlayers option
            self.decoder = L.StatelessLSTM(dunits + eprojs, dunits)  # 310s per 100 ite -> 240s from NStepLSTM
            self.output = L.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.verbose = verbose
        self.char_list = char_list

    def __call__(self, hs, ys):
        '''

        :param hs:
        :param ys:
        :return:
        '''
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
        c = None
        z = None
        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h
        att_weight_all = []  # for debugging

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs, z, att_w)
            ey = F.hstack((eys[:, i, :], att_c))  # utt x (zdim + hdim)
            c, z = self.decoder(c, z, ey)
            z_all.append(z)
            att_weight_all.append(att_w.data)  # for debugging

        z_all = F.reshape(F.stack(z_all, axis=1), (batch * olength, self.dunits))
        # compute loss
        y_all = self.output(z_all)
        self.loss = F.softmax_cross_entropy(y_all, F.concat(pad_ys_out, axis=0))
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = F.accuracy(y_all, F.concat(pad_ys_out, axis=0), ignore_label=-1)
        logging.info('att loss:' + str(self.loss.data))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            y_hat = F.reshape(y_all, (batch, olength, -1))
            y_true = F.reshape(F.concat(pad_ys_out, axis=0), (batch, olength))
            for (i, y_hat_), y_true_ in zip(enumerate(y_hat.data), y_true.data):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = self.xp.argmax(y_hat_[y_true_ != -1], axis=1)
                idx_true = y_true_[y_true_ != -1]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " + seq_true, i)
                logging.info("prediction [%d]: " + seq_hat, i)

        return self.loss, acc, att_weight_all

    def recognize(self, h, recog_args):
        '''

        :param h:
        :param recog_args:
        :return:
        '''
        logging.info('input lengths: ' + str(h.shape[0]))
        # initialization
        c = None
        z = None
        att_w = None
        y_seq = []
        self.att.reset()  # reset pre-computation of h

        # preprate sos
        y = self.xp.full(1, self.sos, 'i')
        maxlen = int(recog_args.maxlenratio * h.shape[0])
        minlen = int(recog_args.minlenratio * h.shape[0])
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))
        for i in six.moves.range(minlen, maxlen):
            ey = self.embed(y)           # utt list (1) x zdim
            att_c, att_w = self.att([h], z, att_w)
            ey = F.hstack((ey, att_c))   # utt(1) x (zdim + hdim)
            c, z = self.decoder(c, z, ey)
            y = self.xp.argmax(self.output(z).data, axis=1).astype('i')
            y_seq.append(y)

            # terminate decoding
            if y == self.eos:
                break

        return y_seq

    def recognize_beam(self, h, recog_args, char_list):
        '''

        :param h:
        :param recog_args:
        :param char_list:
        :return:
        '''
        logging.info('input lengths: ' + str(h.shape[0]))
        # initialization
        c = None
        z = None
        a = None
        self.att.reset()  # reset pre-computation of h

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty

        # preprate sos
        y = self.xp.full(1, self.sos, 'i')
        maxlen = max(1, int(recog_args.maxlenratio * h.shape[0]))  # maxlen >= 1
        minlen = int(recog_args.minlenratio * h.shape[0])
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c, 'z_prev': z, 'a_prev': a}
        hyps = [hyp]
        ended_hyps = []
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                ey = self.embed(hyp['yseq'][i])           # utt list (1) x zdim
                att_c, att_w = self.att([h], hyp['z_prev'], hyp['a_prev'])
                ey = F.hstack((ey, att_c))   # utt(1) x (zdim + hdim)
                c, z = self.decoder(hyp['c_prev'], hyp['z_prev'], ey)
                hyp['c_prev'] = c
                hyp['z_prev'] = z
                hyp['a_prev'] = att_w

                # get nbest local scores and their ids
                local_scores = F.log_softmax(self.output(z)).data
                local_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][:beam]

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['z_prev'] = z
                    new_hyp['c_prev'] = c
                    new_hyp['a_prev'] = att_w
                    new_hyp['score'] = hyp['score'] + local_scores[0, local_best_ids[j]]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = self.xp.full(1, local_best_ids[j], 'i')
                    hyps_best_kept.append(new_hyp)  # will be (2 x beam) hyps at most

                hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            logging.debug('best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.xp.full(1, self.eos, 'i'))

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
            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug('hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        best_hyp = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[0]
        logging.info('total log probability: ' + str(best_hyp['score']))
        logging.info('normalized log probability: ' + str(best_hyp['score'] / len(best_hyp['yseq'])))

        # remove sos
        return best_hyp['yseq'][1:]


# ------------- Encoder Network ----------------------------------------------------------------------------------------
# TODO avoid to use add_link
class Encoder(chainer.Chain):
    '''ENCODER NEWTWORK CLASS

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
        with self.init_scope():
            if etype == 'blstmp':
                self.enc1 = BLSTMP(idim, elayers, eunits, eprojs, subsample, dropout)
                logging.info('BLSTM with every-layer projection for encoder')
            elif etype == 'vggblstmp':
                self.enc1 = VGG2L(in_channel)
                self.enc2 = BLSTMP(_get_vgg2l_odim(idim, in_channel=in_channel), elayers, eunits, eprojs,
                                   subsample, dropout)
                logging.info('Use CNN-VGG + BLSTMP for encoder')
            elif etype == 'vggblstm':
                self.enc1 = VGG2L(in_channel)
                self.enc2 = BLSTM(_get_vgg2l_odim(idim, in_channel=in_channel), elayers, eunits, eprojs, dropout)
                logging.info('Use CNN-VGG + BLSTM for encoder')
            else:
                logging.error("Error: need to specify an appropriate encoder archtecture")
                sys.exit()

        self.etype = etype

    def __call__(self, xs, ilens):
        '''

        :param xs:
        :param ilens:
        :return:
        '''
        if self.etype == 'blstmp':
            xs, ilens = self.enc1(xs, ilens)
        elif self.etype == 'vggblstmp':
            xs, ilens = self.enc1(xs, ilens)
            xs, ilens = self.enc2(xs, ilens)
        elif self.etype == 'vggblstm':
            xs, ilens = self.enc1(xs, ilens)
            xs, ilens = self.enc2(xs, ilens)
        else:
            logging.error("Error: need to specify an appropriate encoder archtecture")
            sys.exit()

        return xs, ilens


class BLSTMP(chainer.Chain):
    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout):
        super(BLSTMP, self).__init__()
        with self.init_scope():
            for i in six.moves.range(elayers):
                if i == 0:
                    inputdim = idim
                else:
                    inputdim = hdim
                self.add_link("bilstm%d" % i, L.NStepBiLSTM(1, inputdim, cdim, dropout))
                # bottleneck layer to merge
                self.add_link("bt%d" % i, L.Linear(2 * cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample

    def __call__(self, xs, ilens):
        '''

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        for layer in six.moves.range(self.elayers):
            hy, cy, ys = self['bilstm' + str(layer)](None, None, xs)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ys, ilens = _subsamplex(ys, self.subsample[layer + 1])
            ys = self['bt' + str(layer)](F.vstack(ys))  # (sum _utt frame_utt) x dim
            xs = F.split_axis(ys, _ilens_to_index(ilens), axis=0)
            del hy, cy

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), _ilens_to_index(ilens), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


class BLSTM(chainer.Chain):
    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(BLSTM, self).__init__()
        with self.init_scope():
            self.nblstm = L.NStepBiLSTM(elayers, idim, cdim, dropout)
            self.l_last = L.Linear(cdim * 2, hdim)

    def __call__(self, xs, ilens):
        '''

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        hy, cy, ys = self.nblstm(None, None, xs)
        ys = self.l_last(F.vstack(ys))  # (sum _utt frame_utt) x dim
        xs = F.split_axis(ys, _ilens_to_index(ilens), axis=0)
        del hy, cy

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), _ilens_to_index(ilens), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


class VGG2L(chainer.Chain):
    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        with self.init_scope():
            # CNN layer (VGG motivated)
            self.conv1_1 = L.Convolution2D(in_channel, 64, 3, stride=1, pad=1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, stride=1, pad=1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, stride=1, pad=1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1)

        self.in_channel = in_channel

    def __call__(self, xs, ilens):
        '''

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        xs = F.pad_sequence(xs)

        # x: utt x 1 (input channel num) x frame x dim
        xs = F.swapaxes(F.reshape(xs, (xs.shape[0], xs.shape[1], self.in_channel, xs.shape[2] / self.in_channel)), 1, 2)

        xs = F.relu(self.conv1_1(xs))
        xs = F.relu(self.conv1_2(xs))
        xs = F.max_pooling_2d(xs, 2, stride=2)

        xs = F.relu(self.conv2_1(xs))
        xs = F.relu(self.conv2_2(xs))
        xs = F.max_pooling_2d(xs, 2, stride=2)

        # change ilens accordingly
        ilens = self.xp.array(self.xp.ceil(self.xp.array(ilens, dtype=np.float32) / 2), dtype=np.int32)
        ilens = self.xp.array(self.xp.ceil(self.xp.array(ilens, dtype=np.float32) / 2), dtype=np.int32)

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = F.swapaxes(xs, 1, 2)
        xs = F.reshape(xs, (xs.shape[0], xs.shape[1], xs.shape[2] * xs.shape[3]))
        xs = [xs[i, :ilens[i], :] for i in range(len(ilens))]

        return xs, ilens
