#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import logging
import sys

import numpy as np
import six

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from e2e_asr_attctc import BLSTM, BLSTMP
from e2e_asr_attctc import linear_tensor


def _flatten(outer_list):
    flens = [len(inner_list) for inner_list in outer_list]
    flatten_list = [item for inner_list in outer_list for item in inner_list]
    return flatten_list, flens


def _deflatten(flatten_list, flens):
    flens = [0, ] + flens
    flens = np.cumsum(flens)
    deflatten_list = [flatten_list[flens[i]:flens[i + 1]] for i in six.moves.range(len(flens) - 1)]
    return deflatten_list


# Get matrices whose diagonal elements are zero, which is applied for the last two dimensions of a tensor
def _diag_zeros(x):
    if x.shape[-2] != x.shape[-1]:
        logging.error(
            "Error: need to match the last two dimensions")
        sys.exit()

    xp = cuda.get_array_module(x)
    E = np.ones(x.shape[-2:], dtype=np.float32)
    E = np.triu(E, k=1) + np.tril(E, k=-1)
    E = chainer.Variable(xp.array(E, dtype=np.float32))

    return F.scale(x, E, axis=len(x.shape) - 2)


def _batch_matmul_complex(a, b, transa, transb):
    # b x m x n | b x n
    a_real = a['real']
    a_imag = a['imag']
    b_real = b['real']
    b_imag = b['imag']

    if (transa is False) and (transb is False):
        o_real = F.batch_matmul(a_real, b_real, transa, transb) \
            - F.batch_matmul(a_imag, b_imag, transa, transb)
        o_imag = F.batch_matmul(a_real, b_imag, transa, transb) \
            + F.batch_matmul(a_imag, b_real, transa, transb)
    elif (transa is False) and (transb is True):
        o_real = F.batch_matmul(a_real, b_real, transa, transb) \
            + F.batch_matmul(a_imag, b_imag, transa, transb)
        o_imag = -F.batch_matmul(a_real, b_imag, transa, transb) \
            + F.batch_matmul(a_imag, b_real, transa, transb)
    elif (transa is True) and (transb is False):
        o_real = F.batch_matmul(a_real, b_real, transa, transb) \
            + F.batch_matmul(a_imag, b_imag, transa, transb)
        o_imag = F.batch_matmul(a_real, b_imag, transa, transb) \
            - F.batch_matmul(a_imag, b_real, transa, transb)
    elif (transa is True) and (transb is True):
        o_real = F.batch_matmul(a_real, b_real, transa, transb) \
            - F.batch_matmul(a_imag, b_imag, transa, transb)
        o_imag = -F.batch_matmul(a_real, b_imag, transa, transb) \
            - F.batch_matmul(a_imag, b_real, transa, transb)

    # convert to dictionary for output
    o = {}
    o['real'] = o_real
    o['imag'] = o_imag

    return o


def _batch_inv_complex(a, coef=1e-10):
    # m x n x n
    a_real = a['real']
    a_imag = a['imag']
    xp = cuda.get_array_module(a_real)

    # n x n
    E = coef * np.eye(a_real.shape[-1], dtype=np.float32)
    E = chainer.Variable(xp.array(E, dtype=np.float32))

    # m x n x n
    a_real = F.bias(a_real, E, axis=1)

    # see "The Matrix Cookbook" (http://www2.imm.dtu.dk/pubdb/p.php?3274)
    # utilized "Section 4.3" only in case of "t=1"
    o_real = F.batch_inv(a_real + F.batch_matmul(F.batch_matmul(a_imag, F.batch_inv(a_real)), a_imag))
    o_imag = -F.batch_matmul(o_real, F.batch_matmul(a_imag, F.batch_inv(a_real)))

    # convert to dictionary for output
    o = {}
    o['real'] = o_real
    o['imag'] = o_imag

    return o


def _batch_trace_complex(a):
    # m x n x n
    a_real = a['real']
    a_imag = a['imag']
    xp = cuda.get_array_module(a_real)

    # n x n
    E = np.eye(a_real.shape[-1], dtype=np.float32)
    E = chainer.Variable(xp.array(E, dtype=np.float32))

    # m x n x n
    o_real = F.scale(a_real, E, axis=1)
    o_imag = F.scale(a_imag, E, axis=1)

    # m
    o_real = F.sum(o_real, axis=(1, 2))
    o_imag = F.sum(o_imag, axis=(1, 2))

    # convert to dictionary for output
    o = {}
    o['real'] = o_real
    o['imag'] = o_imag

    return o


def _batch_divide_complex(a, b):
    # num : m x n x n
    a_real = a['real']
    a_imag = a['imag']

    # den : m
    b_real = b['real']
    b_imag = b['imag']

    # m x n x n
    num_real = F.scale(a_real, b_real, axis=0) + F.scale(a_imag, b_imag, axis=0)
    num_imag = -F.scale(a_real, b_imag, axis=0) + F.scale(a_imag, b_real, axis=0)

    # m
    den = b_real * b_real + b_imag + b_imag

    # m x n x n
    o_real = F.scale(num_real, 1 / den, axis=0)
    o_imag = F.scale(num_imag, 1 / den, axis=0)

    # convert to dictionary for output
    o = {}
    o['real'] = o_real
    o['imag'] = o_imag

    return o


class NB_MVDR(chainer.Chain):
    def __init__(self, bidim, args):
        super(NB_MVDR, self).__init__()
        with self.init_scope():
            self.mask = MaskEstimator(args.btype, bidim, args.blayers, args.bunits, args.bprojs, args.dropout_rate)
            self.ref = AttRef(bidim, args.bunits, args.bprojs, args.badim, args.dropout_rate)

    def __call__(self, data):

        # utt list of channel list of frame x (freq | bprojs)
        masks, hs = self.mask(data)

        # utt list of freq x channel x channel
        psds = {}
        psds['speech'] = self._get_psd_SN(data, masks['speech'])
        psds['noise'] = self._get_psd_SN(data, masks['noise'])

        # utt x channel
        u = self.ref(psds['speech'], hs)

        # utt list of freq x channel
        ws = self._get_filter(psds, u)

        # utt list of frame x freq
        es = self._filtering(data, ws)

        return es

    def _get_psd_SN(self, xs, mask):
        # compute psd matrix for noisy
        # utt list of frame x freq x channel x channel
        psd_Y = self._get_psd_Y(xs)
        psd_Y_real = psd_Y['real']
        psd_Y_imag = psd_Y['imag']

        # compute mean mask
        # utt list of frame x freq
        mask = self._get_mean_mask(mask)

        # store for broadcast_to
        shapes = [x.shape for x in mask]

        # utt list of freq
        sum_mask = [F.sum(utt, axis=0) for utt in mask]
        # utt list of frame x freq
        sum_mask = [F.broadcast_to(utt, shape) for utt, shape in zip(sum_mask, shapes)]
        # utt list of frame x freq
        mask = [num / den for num, den in zip(mask, sum_mask)]

        # store for split_axis
        ilens = [x.shape[0] for x in psd_Y_real]

        # (utt * frame) x freq x channel x channel
        psd_Y_real = F.vstack(psd_Y_real)
        psd_Y_imag = F.vstack(psd_Y_imag)

        # (utt * frame) x freq
        mask = F.vstack(mask)

        # (utt * frame) x freq x channel x channel
        psd_real = F.scale(psd_Y_real, mask, axis=0)
        psd_imag = F.scale(psd_Y_imag, mask, axis=0)

        # utt list of frame x freq x channel x channel
        psd_real = F.split_axis(psd_real, np.cumsum(ilens[:-1]), axis=0)
        psd_imag = F.split_axis(psd_imag, np.cumsum(ilens[:-1]), axis=0)

        # utt list of freq x channel x channel
        psd_real = [F.sum(utt, axis=0) for utt in psd_real]
        psd_imag = [F.sum(utt, axis=0) for utt in psd_imag]

        # convert to dictionary for output
        psd = {}
        psd['real'] = psd_real
        psd['imag'] = psd_imag

        return psd

    def _get_psd_Y(self, xs):
        # utt list of channel list of frame x freq
        xs_real = xs['real']
        xs_imag = xs['imag']

        # utt list of frame x freq x channel
        xs_real = [F.stack(utt, axis=2) for utt in xs_real]
        xs_imag = [F.stack(utt, axis=2) for utt in xs_imag]

        # store for split_axis
        ilens = [x.shape[0] for x in xs_real]

        # (utt * frame) x freq x channel
        xs_real = F.vstack(xs_real)
        xs_imag = F.vstack(xs_imag)

        # store for reshape
        utdim, fdim, cdim = xs_real.shape

        # (utt * frame * freq) x channel
        xs_real = F.reshape(xs_real, (utdim * fdim, cdim))
        xs_imag = F.reshape(xs_imag, (utdim * fdim, cdim))

        # convert to dictionary for input
        xs = {}
        xs['real'] = xs_real
        xs['imag'] = xs_imag

        # (utt * frame * freq) x channel x channel
        psd_Y = _batch_matmul_complex(xs, xs, False, True)
        psd_Y_real = psd_Y['real']
        psd_Y_imag = psd_Y['imag']

        # (utt * frame) x freq x channel x channel
        psd_Y_real = F.reshape(psd_Y_real, (utdim, fdim, cdim, cdim))
        psd_Y_imag = F.reshape(psd_Y_imag, (utdim, fdim, cdim, cdim))

        # utt list of frame x freq x channel x channel
        psd_Y_real = F.split_axis(psd_Y_real, np.cumsum(ilens[:-1]), axis=0)
        psd_Y_imag = F.split_axis(psd_Y_imag, np.cumsum(ilens[:-1]), axis=0)

        # convert to dictionary for output
        psd_Y = {}
        psd_Y['real'] = psd_Y_real
        psd_Y['imag'] = psd_Y_imag

        return psd_Y

    def _get_mean_mask(self, mask):
        # utt list of frame x freq x channel
        mask = [F.stack(utt, axis=2) for utt in mask]
        # utt list of frame x freq
        mask = [F.average(utt, axis=2) for utt in mask]

        return mask

    def _get_filter(self, psds, u):
        # utt list of freq x channel x channel
        # speech
        psd_S_real = psds['speech']['real']
        psd_S_imag = psds['speech']['imag']
        # noise
        psd_N_real = psds['noise']['real']
        psd_N_imag = psds['noise']['imag']

        # store for broadcast_to
        udim = len(psd_S_real)
        fdim, cdim, _ = psd_S_real[0].shape

        # store for split_axis
        flens = [x.shape[0] for x in psd_S_real]

        # (utt * freq) x channel x channel
        # speech
        psd_S_real = F.vstack(psd_S_real)
        psd_S_imag = F.vstack(psd_S_imag)
        # noise
        psd_N_real = F.vstack(psd_N_real)
        psd_N_imag = F.vstack(psd_N_imag)

        # convert to dictionary for input
        # speech
        psd_S = {}
        psd_S['real'] = psd_S_real
        psd_S['imag'] = psd_S_imag
        # noise
        psd_N = {}
        psd_N['real'] = psd_N_real
        psd_N['imag'] = psd_N_imag

        # (psd_N)^{-1}
        # (utt * freq) x channel x channel
        inv_psd_N = _batch_inv_complex(psd_N)

        # num = (psd_N)^{-1} * (psd_S)
        # (utt * freq) x channel x channel
        num = _batch_matmul_complex(inv_psd_N, psd_S, False, False)

        # den = trace((psd_N)^{-1} * (psd_S))
        # (utt * freq)
        den = _batch_trace_complex(num)

        # w = num/den
        # (utt * freq) x channel x channel
        ws = _batch_divide_complex(num, den)

        # utt x 1 x channel
        u = F.expand_dims(u, 1)
        # utt x freq x channel
        u = F.broadcast_to(u, (udim, fdim, cdim))
        # (utt * freq) x channel
        u = F.reshape(u, (udim * fdim, cdim))

        # convert to dictionary for input
        us = {}
        us['real'] = u
        us['imag'] = chainer.Variable(self.xp.array(
            np.zeros(u.shape, dtype=np.float32), dtype=np.float32))

        # compute filter weight
        # (utt * freq) x channel x 1
        ws = _batch_matmul_complex(ws, us, False, False)
        w_real = ws['real']
        w_imag = ws['imag']

        # (utt * freq) x channel
        w_real = F.reshape(w_real, w_real.shape[:-1])
        w_imag = F.reshape(w_imag, w_imag.shape[:-1])

        # utt list of freq x channel
        w_real = F.split_axis(w_real, np.cumsum(flens[:-1]), axis=0)
        w_imag = F.split_axis(w_imag, np.cumsum(flens[:-1]), axis=0)

        # convert to dictionary for output
        ws = {}
        ws['real'] = w_real
        ws['imag'] = w_imag

        return ws

    def _filtering(self, xs, ws):
        # utt list of channel list of frame x freq
        xs_real = xs['real']
        xs_imag = xs['imag']

        # utt list of frame x freq x channel
        xs_real = [F.stack(utt, axis=2) for utt in xs_real]
        xs_imag = [F.stack(utt, axis=2) for utt in xs_imag]

        # utt list of freq x channel
        ws_real = ws['real']
        ws_imag = ws['imag']

        # store for broadcast_to
        shapes = [x.shape for x in xs_real]

        # utt list of frame x freq x channel
        ws_real = [F.broadcast_to(utt, shape) for utt, shape in zip(ws_real, shapes)]
        ws_imag = [F.broadcast_to(utt, shape) for utt, shape in zip(ws_imag, shapes)]

        # store for split_axis
        ilens = [x.shape[0] for x in xs_real]

        # (utt * frame) x freq x channel
        xs_real = F.vstack(xs_real)
        xs_imag = F.vstack(xs_imag)

        # store for reshape
        utdim, fdim, cdim = xs_real.shape

        # (utt * frame * freq) x channel
        xs_real = F.reshape(xs_real, (utdim * fdim, cdim))
        xs_imag = F.reshape(xs_imag, (utdim * fdim, cdim))

        # convert to dictionary for input
        xs = {}
        xs['real'] = xs_real
        xs['imag'] = xs_imag

        # (utt * frame) x freq x channel
        ws_real = F.vstack(ws_real)
        ws_imag = F.vstack(ws_imag)

        # (utt * frame * freq) x channel
        ws_real = F.reshape(ws_real, (utdim * fdim, cdim))
        ws_imag = F.reshape(ws_imag, (utdim * fdim, cdim))

        # convert to dictionary for input
        ws = {}
        ws['real'] = ws_real
        ws['imag'] = ws_imag

        # (utt * frame * freq) x 1 x 1
        hs = _batch_matmul_complex(ws, xs, True, False)
        hs_real = hs['real']
        hs_imag = hs['imag']

        # (utt * frame) x freq
        hs_real = F.reshape(hs_real, (utdim, fdim))
        hs_imag = F.reshape(hs_imag, (utdim, fdim))

        # utt list of frame x freq
        hs_real = F.split_axis(hs_real, np.cumsum(ilens[:-1]), axis=0)
        hs_imag = F.split_axis(hs_imag, np.cumsum(ilens[:-1]), axis=0)

        # convert to dictionary for output
        hs = {}
        hs['real'] = hs_real
        hs['imag'] = hs_imag

        return hs


# Mask estimation network
class MaskEstimator(chainer.Chain):
    '''MASK ESTIMATION NETWORK CLASS

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

    def __init__(self, btype, bidim, blayers, bunits, bprojs, dropout):
        # subsampling is not performed in mask estimation network
        subsample = np.ones(blayers + 1, dtype=np.int)

        super(MaskEstimator, self).__init__()
        with self.init_scope():
            if btype == 'blstm':
                self.blstm = BLSTM(2 * bidim, blayers, bunits, bprojs, dropout)
                logging.info('BLSTM without projection for mask estimator')
            elif btype == 'blstmp':
                self.blstm = BLSTMP(2 * bidim, blayers, bunits,
                                    bprojs, subsample, dropout)
                logging.info('BLSTM with every-layer projection for mask estimator')
            else:
                logging.error(
                    "Error: need to specify an appropriate encoder archtecture")
                sys.exit()
            self.lo_S = L.Linear(bprojs, bidim)
            self.lo_N = L.Linear(bprojs, bidim)

    def __call__(self, xs):
        '''Mask estimator forward

        :param xs:
        :return:
        '''
        # utt list of channel list of frame x 2 freq
        xs = [[F.hstack([ch_real, ch_imag]) for ch_real, ch_imag in zip(utt_real, utt_imag)]
              for utt_real, utt_imag in zip(xs['real'], xs['imag'])]

        # utt channel list of frame x 2 freq
        xs, flens = _flatten(xs)
        ilens = np.array([xx.shape[0] for xx in xs], dtype=np.int32)
        # utt channel list of frame x bprojs
        xs, ilens = self.blstm(xs, ilens)
        # (utt channel * frame) x bprojs
        xs = F.vstack(xs)

        # speech
        ms_S = F.sigmoid(self.lo_S(xs))  # (utt channel * frame) x freq
        ms_S = F.split_axis(ms_S, np.cumsum(ilens[:-1]), axis=0)  # utt channel list of frame x freq
        ms_S = _deflatten(ms_S, flens)  # utt list of channel list of frame x freq
        # noise
        ms_N = F.sigmoid(self.lo_N(xs))  # (utt channel * frame) x freq
        ms_N = F.split_axis(ms_N, np.cumsum(ilens[:-1]), axis=0)  # utt channel list of frame x freq
        ms_N = _deflatten(ms_N, flens)  # utt list of channel list of frame x freq

        # convert to dictionary for output
        ms = {}
        ms['speech'] = ms_S
        ms['noise'] = ms_N

        # utt channel list of frame x bprojs
        xs = F.split_axis(xs, np.cumsum(ilens[:-1]), axis=0)
        # utt list of channel list of frame x bprojs
        xs = _deflatten(xs, flens)

        return ms, xs


# Attention-based reference selection
class AttRef(chainer.Chain):
    def __init__(self, bidim, bunits, bprojs, att_dim, dropout):
        super(AttRef, self).__init__()
        with self.init_scope():
            self.mlp_psd = L.Linear(2 * bidim, att_dim)
            self.mlp_state = L.Linear(bprojs, att_dim, nobias=True)
            self.gvec = L.Linear(att_dim, 1)

    def __call__(self, psd_in, state_in, scaling=2.0):
        # psd_feat
        # utt list of freq x channel x channel
        psd_real = psd_in['real']
        psd_imag = psd_in['imag']
        # utt x freq x channel x channel
        psd_real = F.stack(psd_real, axis=0)
        psd_imag = F.stack(psd_imag, axis=0)
        # utt x freq x channel
        cdim = psd_real.shape[-1]
        psd_real = F.sum(_diag_zeros(psd_real), axis=3) / (cdim - 1)
        psd_imag = F.sum(_diag_zeros(psd_imag), axis=3) / (cdim - 1)
        # utt x channel x freq
        psd_real = F.rollaxis(psd_real, axis=2, start=1)
        psd_imag = F.rollaxis(psd_imag, axis=2, start=1)
        # utt x channel x 2 freq
        psd_feat = F.concat([psd_real, psd_imag], axis=2)

        # state_feat
        # utt list of channel list of frame x bprojs
        state_feat = state_in
        # utt list of frame x bprojs x channel
        state_feat = [F.stack(utt, axis=2) for utt in state_feat]
        # utt list of bprojs x channel
        state_feat = [F.average(utt, axis=0) for utt in state_feat]
        # utt x bprojs x channel
        state_feat = F.stack(state_feat, axis=0)
        # utt x channel x brojs
        state_feat = F.rollaxis(state_feat, axis=2, start=1)

        # dot with gvec
        # utt x channel x att_dim
        mlp_psd = linear_tensor(self.mlp_psd, psd_feat)
        mlp_state = linear_tensor(self.mlp_state, state_feat)
        # utt x channel
        e = F.squeeze(linear_tensor(self.gvec, F.tanh(
            mlp_psd + mlp_state)), axis=2)
        u = F.softmax(scaling * e)

        return u
