"""Parallel beam search module."""
from __future__ import annotations

import logging
from typing import Dict
from typing import List
from typing import NamedTuple

import torch

from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis
from espnet.nets.scorer_interface import ScorerInterface


class BatchHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type."""

    yseq: torch.Tensor
    score: torch.Tensor
    ended: torch.Tensor
    scores: Dict[str, torch.Tensor] = dict()
    states: Dict[str, Dict] = dict()

    def __init__(self, hs: List[Hypothesis], eos=0):
        """Load data from list."""
        raise NotImplementedError

    def to_list(self, eos=0) -> List[Hypothesis]:
        """Convert data to list."""
        raise NotImplementedError

    def select(self, ids, scorers: Dict[str, ScorerInterface]) -> BatchHypothesis:
        """Select new batch from hypothesis ids."""
        raise NotImplementedError


class BatchBeamSearch(BeamSearch):
    """Batch beam search implementation."""

    def score(self, running_hyps: List[Hypothesis], x: torch.Tensor) -> List[Hypothesis]:
        """Score running hypotheses for encoded speech x.

        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            List[Hypotheses]: Best sorted hypotheses

        """
        # TODO(karita): implement
        raise NotImplementedError

        n_batch = len(running_hyps)
        batch_hyps = BatchHypothesis(running_hyps)
        # TODO(karita): implement batch scorer
        scores, states = self.score_full(batch_hyps, x)

        # TODO(karita): implement batch partial scorer
        # part_ids = self.pre_beam(scores, device=x.device)
        # part_scores, part_states = self.score_partial(batch_hyps, part_ids, x)

        # TODO(karita): mask ended hyps
        # weighted sum scores
        weighted_scores = torch.zeros(
            n_batch, self.n_vocab, dtype=x.dtype, device=x.device)
        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]
        # TODO(karita): implement batch partial scorer
        # for k in self.part_scorers:
        #     weighted_scores[part_ids] += self.weights[k] * \
        #         part_scores[k]
        weighted_scores += batch_hyps.score

        # update hyps
        best = []
        for i, hyp in enumerate(running_hyps):
            for j, part_j in zip(*self.main_beam(weighted_scores, part_ids)):
                # will be (2 x beam at most)
                best.append(Hypothesis(
                    score=weighted_scores[i, j],
                    yseq=self.append_token(hyp.yseq, j))
                    # TODO(karita): implement merge op
                    # scores=self.merge_scores(
                    #     hyp, scores, j, part_scores, part_j),
                    # states=self.merge_states(states, part_states, part_j)))

        # sort and prune 2 x beam -> beam
        return self.top_beam_hyps(best)
