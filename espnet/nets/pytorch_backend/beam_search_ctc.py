"""Search algorithms for CTC models."""

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

from espnet.nets.scorer_interface import ScorerInterface

@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms."""

    score: float
    norm_score: float
    yseq: List[int]

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return {
            'yseq': self.yseq,
            'score': self.score,
            'norm_score': self.norm_score
        }

@dataclass
class LMCache:
    """LMCache class for beam search algorithms."""

    state: Any
    scores: Any
    score_sum: float

class BeamSearchCTC(torch.nn.Module):
    """Beam search implementation for frame-synchronous CTC decoding,
       which is similar to [1] but this code has some improvements.
        [1] A. Maas, A. Hannun, D. Jurafsky, A. Ng: First-Pass Large Vocabulary 
            Continuous Speech Recognition using Bi-Directional Recurrent DNNs,
            2014, https://arxiv.org/abs/1408.2873
    """

    def __init__(
        self,
        beam_size: int,
        sos: int,
        ctc: torch.nn.Module,
        blank: int = 0,
        lm: ScorerInterface = None,
        lm_weight: float = 1.0,
        pruning_width: float = 18.0,
        insertion_bonus: float = 1.0,
    ):
        """Initialize frame-synchronous CTC prefix beam search.

        Args:
            beam_size: Number of hypotheses kept during search
            sos: start-of-sentence symbol ID
            ctc: CTC class to use
            blank: blank symbol ID 
            lm: LM class to use
            lm_weight: lm weight for soft fusion
            pruning_width: score-based pruning (score margin: max score difference between best and worst hypothesis)
            insertion_bonus: label insertion bonus weight
        """
        super().__init__()
        assert check_argument_types()
        self.blank = blank
        self.ctc_max_tokens = beam_size
        self.ctc_beam = pruning_width # typically something below 20
        self.ctc_lm_weight = lm_weight  
        self.beta = insertion_bonus      # typically between 0.5 and 2.0
        self.sos = sos
        self.ctc = ctc
        self.lm = lm

        self.local_ctc_threshold = np.log(0.0001)    # local ctc pruning threshold
        self.logzero = float("-inf") #-10000000000.0

        self.lm_cache = dict()
        self.x = None

    #def get_ctc_score(self, l_list: List[Tuple[int]], p_prfx: Dict[Tuple[int], float]):
    def get_ctc_prefix_scores(
        self, l_list: Any, p_prfx: Any
    ) -> Any:
        ctc_scores = dict()
        for l in l_list:
            ctc_score = (1-self.ctc_lm_weight)*np.logaddexp(*p_prfx[l])
            if len(l) > 1 and self.ctc_lm_weight > 0 and self.lm is not None:
                ctc_score += self.score_lm(l)*self.ctc_lm_weight
            ctc_score += self.beta*(len(l)-1)
            ctc_scores[l] = ctc_score
        return ctc_scores

    def init_cache(self, x: torch.Tensor):
        self.reset_cache()
        if self.ctc_lm_weight == 0 or self.lm is None:
            return
        self.x = x
        sos_t = torch.tensor([self.sos], dtype=torch.long)
        init_lm_state = self.lm.init_state(x)
        lm_scores, lm_state = self.lm.score(sos_t, init_lm_state, x)
        self.lm_cache[(self.sos,)] = LMCache(
            state = lm_state,
            scores = lm_scores,
            score_sum = 0.0,
        )

    def reset_cache(self):
        self.lm_cache = dict()
        self.x = None

    def score_lm(
        self, l: Tuple[int], score_sum = True
    ) -> Any:
        root_hyp = l[:-1]
        if root_hyp in self.lm_cache:
            lm_scores = self.lm_cache[root_hyp].scores
            lm_score_sum = self.lm_cache[root_hyp].score_sum
        else:
            pp_hyp = root_hyp[:-1]
            root_lm_state = self.lm_cache[pp_hyp].state
            root_hyp_t = torch.tensor(root_hyp).long()
            lm_scores, lm_state = self.lm.score(root_hyp_t, root_lm_state, self.x)
            lm_score_sum = self.lm_cache[pp_hyp].score_sum
            lm_score_sum += float(self.lm_cache[pp_hyp].scores[root_hyp[-1]])
            self.lm_cache[root_hyp] = LMCache(
                state = lm_state,
                scores = lm_scores,
                score_sum = lm_score_sum
            )
        lm_score = float(lm_scores[l[-1]])
        return lm_score_sum + lm_score if score_sum else lm_score

    def ctc_search_step(
        self, p_ctc: Any, p_prefix: Any, A_pruned: Any
    ) -> Any:
        local_ctc_threshold = self.local_ctc_threshold
        blank = self.blank
        logzero = self.logzero
        pre_beam = int(1.5 * self.ctc_max_tokens)
        local_ctc_threshold = np.sort(p_ctc)[-pre_beam]
        cs_pruned = set(np.where(p_ctc >= local_ctc_threshold)[0])
        # cs_pruned = set(np.where(p_ctc > local_ctc_threshold)[0])
        if len(cs_pruned) == 0:  cs_pruned = {np.argmax(p_ctc)}
        cs_pruned.add(blank) # ensure the blank symbol remains in the set to forward short prefixes until pruned away
        A_next = set()
        p_prefix_next = defaultdict(lambda: (logzero, logzero)) # (p_nb, p_b)
        for l in A_pruned:
            p_prev_l = np.logaddexp(*p_prefix[l])
            for c in cs_pruned:
                if c == blank:
                    p_nb, p_b = p_prefix_next[l]
                    p_b = np.logaddexp(p_b, p_ctc[c] + p_prev_l)
                    if len(l)>1 and l[-1] not in cs_pruned:
                        p_nb = np.logaddexp(p_nb, p_ctc[l[-1]] + p_prefix[l][0])
                    p_prefix_next[l] = (p_nb, p_b)
                    A_next.add(l)
                else:
                    l_plus = l + (int(c),)
                    p_nb, p_b = p_prefix_next[l_plus]
                    if c == l[-1]:
                        p_nb_prev, p_b_prev = p_prefix[l]
                        p_nb = np.logaddexp(p_nb, p_ctc[c] + p_b_prev)
                        p_nb_l, p_b_l = p_prefix_next[l]
                        p_nb_l = np.logaddexp(p_nb_l, p_ctc[c] + p_nb_prev)
                        p_prefix_next[l] = (p_nb_l, p_b_l)
                    else:
                        p_nb = np.logaddexp(p_nb, p_ctc[c] + p_prev_l)
                    if l_plus not in A_pruned and l_plus in p_prefix:
                        p_b = np.logaddexp(p_b, p_ctc[blank] + np.logaddexp(*p_prefix[l_plus]))
                        p_nb = np.logaddexp(p_nb, p_ctc[c] + p_prefix[l_plus][0])
                    p_prefix_next[l_plus] = (p_nb, p_b)
                    A_next.add(l_plus)

        ctc_scores = self.get_ctc_prefix_scores(A_next, p_prefix_next)
        A_pruned = sorted(A_next, key=lambda l:ctc_scores[l], reverse=True)
        A_pruned = A_pruned[:self.ctc_max_tokens]
        A_pruned = [l for l in A_pruned if ctc_scores[l]>ctc_scores[A_pruned[0]]-self.ctc_beam]
        p_prefix = p_prefix_next.copy()
        return p_prefix, A_pruned

    def search(self, enc_output: torch.Tensor) -> List[Hypothesis]:
        """Frame-synchronous CTC beam search

        Args:
            enc_output (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            list[Hypothesis]: N-best decoding results
        """
        # import pdb;pdb.set_trace()
        logging.info("decoder input length: " + str(enc_output.shape[1]))
        lpz = self.ctc.log_softmax(enc_output)
        lpz = lpz.squeeze(0)
        lpz = lpz.cpu().detach().numpy()
        self.init_cache(enc_output)

        A_pruned = [(self.sos,)]
        p_prefix = defaultdict(lambda: (self.logzero, self.logzero)) # (p_nb, p_b)
        p_prefix[(self.sos,)] = (self.logzero, 0.0)
        for t in range(lpz.shape[0]):
            # CTC prefix beam search
            p_prefix, A_pruned = self.ctc_search_step(lpz[t,:], p_prefix, A_pruned)

        final_hyps = []
        ctc_scores = self.get_ctc_prefix_scores(A_pruned, p_prefix)
        for l in A_pruned:
            hyp = Hypothesis(
                yseq = list(l),
                score = ctc_scores[l],
                norm_score = ctc_scores[l] / max(len(l)-1,1),
            )
            final_hyps.append(hyp)
        final_hyps = sorted(final_hyps, key=lambda x: x.score, reverse=True)
        final_hyps = [{"score": h.score, "yseq": h.yseq + [self.sos]} for h in final_hyps]
        best_score = final_hyps[0]["score"]
        logging.info(f"total log probability: {best_score:.2f}")
        return final_hyps
