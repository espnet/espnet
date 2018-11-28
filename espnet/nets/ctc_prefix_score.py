#!/usr/bin/env python

# Copyright 2018 Mitsubishi Electric Research Labs (Takaaki Hori)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

import numpy as np
import six


class CTCPrefixScoreTH(object):
    """Batch processing of CTCPrefixScore

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probablities of multiple labels
    simultaneously
    """

    def __init__(self, x, blank, eos, beam, hlens, device_id):
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.batch = x.size(0)
        self.input_length = x.size(1)
        self.odim = x.size(2)
        self.beam = beam
        self.n_bb = self.batch * beam

        self.hlens = hlens

        self.x = x
        self.device_id = device_id

    def to_cuda(self, x):
        if self.device_id == -1:
            return x
        return x.cuda(self.device_id)

    def initial_state(self):
        """Obtain an initial CTC state

        :return: CTC state
        """
        self.x = self.x.view(self.batch, 1, self.input_length, self.odim)
        self.x = self.x.repeat(1, self.beam, 1, 1)
        self.x = self.x.view(self.n_bb, self.input_length, self.odim)
        self.cs = torch.from_numpy(np.arange(self.odim, dtype=np.int32))
        self.cs = self.to_cuda(self.cs)

        # initial CTC state is made of a n_bb x frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        r = torch.full((self.n_bb, self.input_length, 2), self.logzero)
        r = self.to_cuda(r)

        r[:, 0, 1] = self.x[:, 0, self.blank]
        for i in six.moves.range(1, self.input_length):
            r[:, i, 1] = r[:, i - 1, 1] + self.x[:, i, self.blank]

        self.hlens = [x - 1 for x in self.hlens]

        return r

    def __call__(self, y, r_prev, last=None):
        """Compute CTC prefix scores for next labels

        :param y     : prefix label sequence
        :param cs    : array of next labels
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """

        output_length = len(y[0]) - 1  # ignore sos

        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).
        r = torch.full((self.n_bb, self.input_length, 2, self.odim), self.logzero)
        r = self.to_cuda(r)
        if output_length == 0:
            r[:, 0, 0, :] = self.x[:, 0]

        r_sum = torch.logsumexp(r_prev, dim=2)
        if last is None:
            last = [yi[-1] for yi in y]

        log_phi = r_sum.unsqueeze(2).repeat(1, 1, self.odim)
        log_phi = self.to_cuda(log_phi)
        for idx in six.moves.range(self.n_bb):
            log_phi[idx, :, last[idx]] = r_prev[idx, :, 1]

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilites log(psi)
        start = max(output_length, 1)
        log_psi = r[:, start - 1, 0, :]
        log_phi_x = torch.cat((log_phi[:, 0].unsqueeze(1), log_phi[:, :-1]), dim=1) + self.x
        for t in six.moves.range(start, self.input_length):
            xt = self.x[:, t]
            rp = r[:, t - 1]
            r[:, t, 0] = torch.logsumexp(torch.stack([rp[:, 0], log_phi[:, t - 1]]), dim=0) + xt
            r[:, t, 1] = torch.logsumexp(rp, dim=1) + xt[:, self.blank].view(-1, 1).repeat(1, self.odim)
            log_psi = torch.logsumexp(torch.stack([log_psi, log_phi_x[:, t]]), dim=0)

        for si in six.moves.range(self.n_bb):
            log_psi[si, self.eos] = r_sum[si, self.hlens[si]]

        return log_psi, r


class CTCPrefixScore(object):
    """Compute CTC label sequence scores

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probablities of multiple labels
    simultaneously
    """

    def __init__(self, x, blank, eos, xp):
        self.xp = xp
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.input_length = len(x)
        self.x = x

    def initial_state(self):
        """Obtain an initial CTC state

        :return: CTC state
        """
        # initial CTC state is made of a frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        r = self.xp.full((self.input_length, 2), self.logzero, dtype=np.float32)
        r[0, 1] = self.x[0, self.blank]
        for i in six.moves.range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        return r

    def __call__(self, y, cs, r_prev):
        """Compute CTC prefix scores for next labels

        :param y     : prefix label sequence
        :param cs    : array of next labels
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """
        # initialize CTC states
        output_length = len(y) - 1  # ignore sos
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).
        r = self.xp.ndarray((self.input_length, 2, len(cs)), dtype=np.float32)
        xs = self.x[:, cs]
        if output_length == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.logzero
        else:
            r[output_length - 1] = self.logzero

        # prepare forward probabilities for the last label
        r_sum = self.xp.logaddexp(r_prev[:, 0], r_prev[:, 1])  # log(r_t^n(g) + r_t^b(g))
        last = y[-1]
        if output_length > 0 and last in cs:
            log_phi = self.xp.ndarray((self.input_length, len(cs)), dtype=np.float32)
            for i in six.moves.range(len(cs)):
                log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
        else:
            log_phi = r_sum

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilites log(psi)
        start = max(output_length, 1)
        log_psi = r[start - 1, 0]
        for t in six.moves.range(start, self.input_length):
            r[t, 0] = self.xp.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = self.xp.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.x[t, self.blank]
            log_psi = self.xp.logaddexp(log_psi, log_phi[t - 1] + xs[t])

        # get P(...eos|X) that ends with the prefix itself
        eos_pos = self.xp.where(cs == self.eos)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]  # log(r_T^n(g) + r_T^b(g))

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        return log_psi, self.xp.rollaxis(r, 2)
