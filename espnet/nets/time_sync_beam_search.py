"""
Time Synchronous One-Pass Beam Search.
Implements joint CTC/attention decoding where
hypotheses are expanded along the time (input) axis,
as described in https://arxiv.org/abs/2210.05200.
Supports CPU and GPU inference.
References: https://arxiv.org/abs/1408.2873 for CTC beam search
Author: Brian Yan
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from espnet.nets.beam_search import Hypothesis
from espnet.nets.scorer_interface import ScorerInterface


@dataclass
class CacheItem:
    """For caching attentional decoder and LM states."""

    state: Any
    scores: Any
    log_sum: float


class TimeSyncBeamSearch(torch.nn.Module):
    """Time synchronous beam search algorithm."""

    def __init__(
        self,
        sos: int,
        beam_size: int,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
        token_list=dict,
        pre_beam_ratio: float = 1.5,
        blank: int = 0,
        force_lid: bool = False,
        temp: float = 1.0,
    ):
        """Initialize beam search.

        Args:
            beam_size: num hyps
            sos: sos index
            ctc: CTC module
            pre_beam_ratio: pre_beam_ratio * beam_size = pre_beam
                pre_beam is used to select candidates from vocab to extend hypotheses
            decoder: decoder ScorerInterface
            ctc_weight: ctc_weight
            blank: blank index

        """
        super().__init__()
        self.ctc = scorers["ctc"]
        self.decoder = scorers["decoder"]
        self.lm = scorers["lm"] if "lm" in scorers else None
        self.beam_size = beam_size
        self.pre_beam_size = int(pre_beam_ratio * beam_size)
        self.ctc_weight = weights["ctc"]
        self.lm_weight = weights["lm"]
        self.decoder_weight = weights["decoder"]
        self.penalty = weights["length_bonus"]
        self.sos = sos
        self.sos_th = torch.tensor([self.sos])
        self.blank = blank
        self.attn_cache = dict()  # cache for p_attn(Y|X)
        self.lm_cache = dict()  # cache for p_lm(Y)
        self.enc_output = None  # log p_ctc(Z|X)
        self.force_lid = force_lid
        self.temp = temp
        self.token_list = token_list

    def reset(self, enc_output: torch.Tensor):
        """Reset object for a new utterance."""
        self.attn_cache = dict()
        self.lm_cache = dict()
        self.enc_output = enc_output
        self.sos_th = self.sos_th.to(enc_output.device)

        if self.decoder is not None:
            init_decoder_state = self.decoder.init_state(enc_output)
            decoder_scores, decoder_state = self.decoder.score(
                self.sos_th, init_decoder_state, enc_output
            )
            self.attn_cache[(self.sos,)] = CacheItem(
                state=decoder_state,
                scores=decoder_scores,
                log_sum=0.0,
            )
        if self.lm is not None:
            init_lm_state = self.lm.init_state(enc_output)
            lm_scores, lm_state = self.lm.score(self.sos_th, init_lm_state, enc_output)
            self.lm_cache[(self.sos,)] = CacheItem(
                state=lm_state,
                scores=lm_scores,
                log_sum=0.0,
            )

    def cached_score(self, h: Tuple[int], cache: dict, scorer: ScorerInterface) -> Any:
        """Retrieve decoder/LM scores which may be cached."""
        root = h[:-1]  # prefix
        if root in cache:
            root_scores = cache[root].scores
            root_state = cache[root].state
            root_log_sum = cache[root].log_sum
        else:  # run decoder fwd one step and update cache
            root_root = root[:-1]
            root_root_state = cache[root_root].state
            root_scores, root_state = scorer.score(
                torch.tensor(root, device=self.enc_output.device).long(),
                root_root_state,
                self.enc_output,
            )
            root_log_sum = cache[root_root].log_sum + float(
                cache[root_root].scores[root[-1]]
            )
            cache[root] = CacheItem(
                state=root_state, scores=root_scores, log_sum=root_log_sum
            )
        cand_score = float(root_scores[h[-1]])
        score = root_log_sum + cand_score

        return score

    def joint_score(self, hyps: Any, ctc_score_dp: Any) -> Any:
        """Calculate joint score for hyps."""
        scores = dict()
        for h in hyps:
            score = self.ctc_weight * np.logaddexp(*ctc_score_dp[h])  # ctc score
            if len(h) > 1 and self.decoder_weight > 0 and self.decoder is not None:
                score += (
                    self.cached_score(h, self.attn_cache, self.decoder)
                    * self.decoder_weight
                )  # attn score
            if len(h) > 1 and self.lm is not None and self.lm_weight > 0:
                score += (
                    self.cached_score(h, self.lm_cache, self.lm) * self.lm_weight
                )  # lm score
            score += self.penalty * (len(h) - 1)  # penalty score
            scores[h] = score
        return scores

    def time_step(self, p_ctc: Any, ctc_score_dp: Any, hyps: Any) -> Any:
        """Execute a single time step."""
        pre_beam_threshold = np.sort(p_ctc)[-self.pre_beam_size]
        cands = set(np.where(p_ctc >= pre_beam_threshold)[0])
        if len(cands) == 0:
            cands = {np.argmax(p_ctc)}
        new_hyps = set()
        ctc_score_dp_next = defaultdict(
            lambda: (float("-inf"), float("-inf"))
        )  # (p_nb, p_b)
        tmp = []
        for hyp_l in hyps:
            p_prev_l = np.logaddexp(*ctc_score_dp[l])
            for c in cands:
                if c == self.blank:
                    logging.debug("blank cand, hypothesis is " + str(l))
                    p_nb, p_b = ctc_score_dp_next[hyp_l]
                    p_b = np.logaddexp(p_b, p_ctc[c] + p_prev_l)
                    ctc_score_dp_next[hyp_l] = (p_nb, p_b)
                    new_hyps.add(hyp_l)
                else:
                    l_plus = hyp_l + (int(c),)
                    logging.debug("non-blank cand, hypothesis is " + str(l_plus))
                    p_nb, p_b = ctc_score_dp_next[l_plus]
                    if c == hyp_l[-1]:
                        logging.debug("repeat cand, hypothesis is " + str(l))
                        p_nb_prev, p_b_prev = ctc_score_dp[hyp_l]
                        p_nb = np.logaddexp(p_nb, p_ctc[c] + p_b_prev)
                        p_nb_l, p_b_l = ctc_score_dp_next[hyp_l]
                        p_nb_l = np.logaddexp(p_nb_l, p_ctc[c] + p_nb_prev)
                        ctc_score_dp_next[hyp_l] = (p_nb_l, p_b_l)
                    else:
                        p_nb = np.logaddexp(p_nb, p_ctc[c] + p_prev_l)
                    if l_plus not in hyps and l_plus in ctc_score_dp:
                        p_b = np.logaddexp(
                            p_b, p_ctc[self.blank] + np.logaddexp(*ctc_score_dp[l_plus])
                        )
                        p_nb = np.logaddexp(p_nb, p_ctc[c] + ctc_score_dp[l_plus][0])
                        tmp.append(l_plus)
                    ctc_score_dp_next[l_plus] = (p_nb, p_b)
                    new_hyps.add(l_plus)

        scores = self.joint_score(new_hyps, ctc_score_dp_next)

        hyps = sorted(new_hyps, key=lambda l: scores[l], reverse=True)[: self.beam_size]
        ctc_score_dp = ctc_score_dp_next.copy()
        return ctc_score_dp, hyps, scores

    def forward(
        self, x: torch.Tensor, maxlenratio: float = 0.0, minlenratio: float = 0.0
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            enc_output (torch.Tensor)

        Return:
            list[Hypothesis]

        """
        logging.info("decoder input lengths: " + str(x.shape[0]))
        lpz = self.ctc.log_softmax(x.unsqueeze(0))
        lpz = lpz.squeeze(0)
        lpz = lpz.cpu().detach().numpy()
        self.reset(x)

        hyps = [(self.sos,)]
        ctc_score_dp = defaultdict(
            lambda: (float("-inf"), float("-inf"))
        )  # (p_nb, p_b) - dp object tracking p_ctc
        ctc_score_dp[(self.sos,)] = (float("-inf"), 0.0)
        for t in range(lpz.shape[0]):
            logging.debug("position " + str(t))
            ctc_score_dp, hyps, scores = self.time_step(lpz[t, :], ctc_score_dp, hyps)

        ret = [
            Hypothesis(yseq=torch.tensor(list(h) + [self.sos]), score=scores[h])
            for h in hyps
        ]
        best_hyp = "".join([self.token_list[x] for x in ret[0].yseq.tolist()])
        best_hyp_len = len(ret[0].yseq)
        best_score = ret[0].score
        logging.info(f"output length: {best_hyp_len}")
        logging.info(f"total log probability: {best_score:.2f}")
        logging.info(f"best hypo: {best_hyp}")

        return ret
