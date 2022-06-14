"""
Time Synchronous One-Pass Beam Search -
Implements joint CTC/attention decoding where
hypotheses are expanded along the time (input) axis.

References: https://arxiv.org/abs/1408.2873 for CTC beam search

Author: Brian Yan
"""

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types

from collections import defaultdict
import logging

from espnet.nets.beam_search import Hypothesis
from espnet.nets.scorer_interface import ScorerInterface

@dataclass
class CacheItem:
    """For caching attentional decoder."""
    state: Any
    scores: Any
    log_sum: float

class TimeSyncBeamSearch(torch.nn.Module):
    """Time synchronous beam search algorithm"""

    def __init__(
        self,
        sos: int,
        beam_size: int,
        ctc: torch.nn.Module,
        pre_beam_ratio: float=1.5,
        decoder: ScorerInterface=None,
        ctc_weight: float=1.0,
        penalty: float=1.0,
        blank: int=0,
    ):
        """
        beam_size: num hyps
        sos: sos index
        ctc: CTC module
        pre_beam_ratio: pre_beam_ratio * beam_size = pre_beam
        decoder: decoder ScorerInterface
        ctc_weight: ctc_weight
        blank: blank index
        """
        super().__init__()
        assert check_argument_types()
        self.ctc = ctc
        self.decoder = decoder
        self.beam_size = beam_size
        self.pre_beam_size = int(pre_beam_ratio * beam_size)
        self.ctc_weight = ctc_weight
        self.decoder_weight = 1.0 - ctc_weight  
        self.penalty = penalty
        self.sos = sos
        self.sos_th = torch.tensor([self.sos])
        self.blank = blank
        self.seen_prefixes = dict()     # cache for p_attn(Y|X)
        self.enc_output = None           # log p_ctc(Z|X)

    def reset(self, enc_output: torch.Tensor):
        self.seen_prefixes = dict()
        self.enc_output = enc_output

        init_decoder_state = self.decoder.init_state(enc_output)
        decoder_scores, decoder_state = self.decoder.score(self.sos_th, init_decoder_state, enc_output)
        self.seen_prefixes[(self.sos,)] = CacheItem(
            state = decoder_state,
            scores = decoder_scores,
            log_sum = 0.0,
        )

    def attn_score(
        self, h: Tuple[int], do_eos: bool=False
    ) -> Any:
        root = h[:-1]   #prefix
        if root in self.seen_prefixes:
            root_decoder_scores = self.seen_prefixes[root].scores
            root_decoder_state = self.seen_prefixes[root].state
            root_log_sum = self.seen_prefixes[root].log_sum 
        else:   #run decoder fwd one step and update cache
            root_root = root[:-1]
            root_root_decoder_state = self.seen_prefixes[root_root].state
            root_decoder_scores, root_decoder_state = self.decoder.score(torch.tensor(root).long(), root_root_decoder_state, self.enc_output)
            root_log_sum = self.seen_prefixes[root_root].log_sum + float(self.seen_prefixes[root_root].scores[root[-1]])
            self.seen_prefixes[root] = CacheItem(
                state = root_decoder_state,
                scores = root_decoder_scores,
                log_sum = root_log_sum
            )
        cand_score = float(root_decoder_scores[h[-1]])
        score = root_log_sum + cand_score

        if do_eos:
            decoder_scores, _ = self.decoder.score(torch.tensor(h).long(), root_decoder_state, self.enc_output)
            score = root_log_sum + cand_score + float(decoder_scores[self.sos])
        return score

    def joint_score(
        self, hyps: Any, ctc_score_dp: Any, do_eos: bool=False
    ) -> Any:
        scores = dict()
        for h in hyps:
            score = self.ctc_weight*np.logaddexp(*ctc_score_dp[h])    #ctc score
            if len(h) > 1 and self.decoder_weight > 0 and self.decoder is not None:
                score += self.attn_score(h, do_eos)*self.decoder_weight #attn score
            score += self.penalty*(len(h)-1)    #penalty score
            scores[h] = score
        return scores

    def joint_score_eos(
        self, hyps: Any, ctc_score_dp: Any
    ) -> Any:
        scores = dict()
        for h in hyps:
            score = self.ctc_weight*np.logaddexp(*ctc_score_dp[h])    #ctc score
            if len(h) > 1 and self.decoder_weight > 0 and self.decoder is not None:
                h_eos = h + (self.sos,)
                score += self.attn_score(h_eos)*self.decoder_weight #attn score
            score += self.penalty*(len(h)-1)    #penalty score
            scores[h] = score
        return scores

    def time_step(
        self, p_ctc: Any, ctc_score_dp: Any, hyps: Any, do_eos: bool=False
    ) -> Any:
        pre_beam_threshold = np.sort(p_ctc)[-self.pre_beam_size]
        cands = set(np.where(p_ctc >= pre_beam_threshold)[0])
        if len(cands) == 0:  cands = {np.argmax(p_ctc)}
        # cands.add(self.blank) # ensure the blank symbol remains in the set to forward short prefixes until pruned away
        new_hyps = set()
        ctc_score_dp_next = defaultdict(lambda: (float("-inf"), float("-inf"))) # (p_nb, p_b)
        for l in hyps:
            p_prev_l = np.logaddexp(*ctc_score_dp[l])
            for c in cands:
                if c == self.blank:
                    p_nb, p_b = ctc_score_dp_next[l]
                    p_b = np.logaddexp(p_b, p_ctc[c] + p_prev_l)
                    # if len(l)>1 and l[-1] not in cands:
                    #     p_nb = np.logaddexp(p_nb, p_ctc[l[-1]] + ctc_score_dp[l][0])
                    ctc_score_dp_next[l] = (p_nb, p_b)
                    new_hyps.add(l)
                else:
                    l_plus = l + (int(c),)
                    p_nb, p_b = ctc_score_dp_next[l_plus]
                    if c == l[-1]:
                        p_nb_prev, p_b_prev = ctc_score_dp[l]
                        p_nb = np.logaddexp(p_nb, p_ctc[c] + p_b_prev)
                        p_nb_l, p_b_l = ctc_score_dp_next[l]
                        p_nb_l = np.logaddexp(p_nb_l, p_ctc[c] + p_nb_prev)
                        ctc_score_dp_next[l] = (p_nb_l, p_b_l)
                    else:
                        p_nb = np.logaddexp(p_nb, p_ctc[c] + p_prev_l)
                    if l_plus not in hyps and l_plus in ctc_score_dp:
                        p_b = np.logaddexp(p_b, p_ctc[self.blank] + np.logaddexp(*ctc_score_dp[l_plus]))
                        p_nb = np.logaddexp(p_nb, p_ctc[c] + ctc_score_dp[l_plus][0])
                    ctc_score_dp_next[l_plus] = (p_nb, p_b)
                    new_hyps.add(l_plus)

        
        scores = self.joint_score(new_hyps, ctc_score_dp_next, do_eos)

        hyps = sorted(new_hyps, key=lambda l:scores[l], reverse=True)[:self.beam_size]
        ctc_score_dp = ctc_score_dp_next.copy()
        return ctc_score_dp, hyps, scores

    def search(self, enc_output: torch.Tensor) -> List[Hypothesis]:
        """
        Params:
            enc_output (torch.Tensor)
        Return:
            list[Hypothesis]
        """
        logging.info("input length: " + str(enc_output.shape[1]))
        lpz = self.ctc.log_softmax(enc_output)
        lpz = lpz.squeeze(0)
        lpz = lpz.cpu().detach().numpy()
        self.reset(enc_output)

        hyps = [(self.sos,)]
        ctc_score_dp = defaultdict(lambda: (float("-inf"), float("-inf"))) # (p_nb, p_b) - dp object tracking p_ctc
        ctc_score_dp[(self.sos,)] = (float("-inf"), 0.0)
        for t in range(lpz.shape[0]):
            ctc_score_dp, hyps, scores = self.time_step(lpz[t,:], ctc_score_dp, hyps, do_eos=(t==lpz.shape[0]-1))

        ret = [Hypothesis(yseq=torch.tensor(list(h)+[self.sos]), score=scores[h]) for h in hyps]
        best_hyp = " ".join([str(x) for x in ret[0].yseq.tolist()])
        best_hyp_len = len(ret[0].yseq)
        best_score = ret[0].score
        logging.info(f"output length: {best_hyp_len}")
        logging.info(f"best hyp: {best_hyp}")
        logging.info(f"total log probability: {best_score:.2f}")

        return ret
        # return [{"score": h.score, "yseq": h.yseq} for h in ret]
