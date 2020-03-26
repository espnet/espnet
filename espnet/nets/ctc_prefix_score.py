#!/usr/bin/env python3

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

    def __init__(self, x, xlens, blank, eos, beam, scoring_ratio=1.5, margin=0):
        """Construct CTC prefix scorer

        :param torch.Tensor x: input label posterior sequences (B, T, O)
        :param torch.Tensor xlens: input lengths (B,)
        :param int blank: blank label id
        :param int eos: end-of-sequence id
        :param int beam: beam size
        :param float scoring_ratio: ratio of #scored hypos to beam size
        :param int margin: margin parameter for windowing (0 means no windowing)
        """
        # In the comment lines, we assume T: input_length, B: batch size, W: beam width, O: output dim.
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.batch = x.size(0)
        self.input_length = x.size(1)
        self.odim = x.size(2)
        self.beam = beam
        self.n_bb = self.batch * beam
        self.device = torch.device('cuda:%d' % x.get_device()) if x.is_cuda else torch.device('cpu')
        # Pad the rest of posteriors in the batch
        # TODO(takaaki-hori): need a better way without for-loops
        for i, l in enumerate(xlens):
            if l < self.input_length:
                x[i, l:, :] = self.logzero
                x[i, l:, blank] = 0
        # Set the number of scoring hypotheses (scoring_num=0 means all)
        self.scoring_num = int(beam * scoring_ratio)
        if self.scoring_num >= self.odim:
            self.scoring_num = 0
        # Expand input posteriors for fast computation
        if self.scoring_num == 0:
            xn = x.transpose(0, 1).unsqueeze(2).repeat(1, 1, beam, 1).view(-1, self.n_bb, self.odim)
        else:
            xn = x.transpose(0, 1)
        xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
        self.x = torch.stack([xn, xb])  # (2, T, B, O) or (2, T, BW, O)
        # Setup CTC windowing
        self.margin = margin
        if margin > 0:
            self.frame_ids = torch.arange(self.input_length, dtype=torch.float32, device=self.device)
        # Precompute end frames (BW,)
        self.end_frames = (torch.as_tensor(xlens) - 1).view(self.batch, 1).repeat(1, beam).view(-1)
        # Precompute base indices to convert label ids to corresponding element indices
        self.pad_b = (torch.arange(self.batch, device=self.device) * beam).view(-1, 1)
        self.pad_bo = (torch.arange(self.batch, device=self.device) * (beam * self.odim)).view(-1, 1)
        self.pad_o = (torch.arange(self.batch, device=self.device) * self.odim).unsqueeze(1).repeat(1, beam).view(-1, 1)
        self.bb_idx = torch.arange(self.n_bb, device=self.device).view(-1, 1)

    def __call__(self, y, state, pre_scores=None, att_w=None):
        """Compute CTC prefix scores for next labels

        :param list y: prefix label sequences
        :param tuple state: previous CTC state
        :param torch.Tensor pre_scores: scores for pre-selection of hypotheses (BW, O)
        :param torch.Tensor att_w: attention weights to decide CTC window
        :return new_state, ctc_local_scores (BW, O)
        """
        output_length = len(y[0]) - 1  # ignore sos
        last_ids = [yi[-1] for yi in y]  # last output label ids
        # prepare state info
        if state is None:
            if self.scoring_num > 0:
                r_prev = torch.full((self.input_length, 2, self.batch, self.beam),
                                    self.logzero, dtype=torch.float32, device=self.device)
                r_prev[:, 1] = torch.cumsum(self.x[0, :, :, self.blank], 0).unsqueeze(2)
                r_prev = r_prev.view(-1, 2, self.n_bb)
            else:
                r_prev = torch.full((self.input_length, 2, self.n_bb),
                                    self.logzero, dtype=torch.float32, device=self.device)
                r_prev[:, 1] = torch.cumsum(self.x[0, :, :, self.blank], 0)
            s_prev = 0.0
            f_min_prev = 0
            f_max_prev = 1
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

        # select input dimensions for scoring
        if self.scoring_num > 0 and pre_scores is not None:
            pre_scores[:, self.blank] = self.logzero  # ignore blank from pre-selection
            scoring_ids = torch.topk(pre_scores, self.scoring_num, 1)[1]
            scoring_idmap = torch.full((self.n_bb, self.odim), -1, dtype=torch.long, device=self.device)
            snum = scoring_ids.size(1)
            scoring_idmap[self.bb_idx, scoring_ids] = torch.arange(snum, device=self.device)
            scoring_idx = (scoring_ids + self.pad_o).view(-1)
            x_ = torch.index_select(self.x.view(2, -1, self.batch * self.odim),
                                    2, scoring_idx).view(2, -1, self.n_bb, snum)
        else:
            scoring_ids = None
            scoring_idmap = None
            snum = self.odim
            x_ = self.x

        # new CTC forward probs are prepared as a (T x 2 x BW x S) tensor
        # that corresponds to r_t^n(h) and r_t^b(h) in a batch.
        r = torch.full((self.input_length, 2, self.n_bb, snum),
                       self.logzero, dtype=torch.float32, device=self.device)
        if output_length == 0:
            r[0, 0] = x_[0, 0]

        r_sum = torch.logsumexp(r_prev, 1)
        log_phi = r_sum.unsqueeze(2).repeat(1, 1, snum)
        if scoring_ids is not None:
            for idx in range(self.n_bb):
                pos = scoring_idmap[idx, last_ids[idx]]
                if pos >= 0:
                    log_phi[:, idx, pos] = r_prev[:, 1, idx]
        else:
            for idx in range(self.n_bb):
                log_phi[:, idx, last_ids[idx]] = r_prev[:, 1, idx]

        # decide start and end frames based on attention weights
        if att_w is not None and self.margin > 0:
            f_arg = torch.matmul(att_w, self.frame_ids)
            f_min = max(int(f_arg.min().cpu()), f_min_prev)
            f_max = max(int(f_arg.max().cpu()), f_max_prev)
            start = min(f_max_prev, max(f_min - self.margin, output_length, 1))
            end = min(f_max + self.margin, self.input_length)
        else:
            f_min = f_max = 0
            start = max(output_length, 1)
            end = self.input_length

        # compute forward probabilities log(r_t^n(h)) and log(r_t^b(h))
        for t in range(start, end):
            rp = r[t - 1]
            rr = torch.stack([rp[0], log_phi[t - 1], rp[0], rp[1]]).view(2, 2, self.n_bb, snum)
            r[t] = torch.logsumexp(rr, 1) + x_[:, t]

        # compute log prefix probabilites log(psi)
        log_phi_x = torch.cat((log_phi[0].unsqueeze(0), log_phi[:-1]), dim=0) + x_[0]
        if scoring_ids is not None:
            log_psi = torch.full((self.n_bb, self.odim), self.logzero, device=self.device)
            log_psi_ = torch.logsumexp(torch.cat((log_phi_x[start:end], r[start - 1, 0].unsqueeze(0)), dim=0), dim=0)
            for si in range(self.n_bb):
                log_psi[si, scoring_ids[si]] = log_psi_[si]
        else:
            log_psi = torch.logsumexp(torch.cat((log_phi_x[start:end], r[start - 1, 0].unsqueeze(0)), dim=0), dim=0)

        for si in range(self.n_bb):
            log_psi[si, self.eos] = r_sum[self.end_frames[si], si]

        # exclude blank probs
        log_psi[:, self.blank] = self.logzero

        return (r, log_psi, f_min, f_max, scoring_idmap), log_psi - s_prev

    def index_select_state(self, state, best_ids):
        """Select CTC states according to best ids

        :param state    : CTC state
        :param best_ids : index numbers selected by beam pruning (B, W)
        :return selected_state
        """
        r, s, f_min, f_max, scoring_idmap = state
        # convert ids to BWO space
        vidx = (best_ids + self.pad_bo).view(-1)
        # select hypothesis scores
        s_new = torch.index_select(s.view(-1), 0, vidx)
        s_new = s_new.view(-1, 1).repeat(1, self.odim).view(self.n_bb, self.odim)
        # convert ids to BWS space (S: scoring_num)
        if scoring_idmap is not None:
            snum = self.scoring_num
            beam_idx = (torch.div(best_ids, self.odim) + self.pad_b).view(-1)
            label_ids = torch.fmod(best_ids, self.odim).view(-1)
            score_idx = scoring_idmap[beam_idx, label_ids]
            score_idx[score_idx == -1] = 0
            vidx = score_idx + beam_idx * snum
        else:
            snum = self.odim
        # select forward probabilities
        r_new = torch.index_select(r.view(-1, 2, self.n_bb * snum), 2, vidx).view(-1, 2, self.n_bb)
        return r_new, s_new, f_min, f_max


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

        # exclude blank probs
        blank_pos = self.xp.where(cs == self.blank)[0]
        if len(blank_pos) > 0:
            log_psi[blank_pos] = self.logzero

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        return log_psi, self.xp.rollaxis(r, 2)
