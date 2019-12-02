"""Parallel beam search module."""

import logging
from typing import Dict
from typing import List
from typing import NamedTuple

import torch
from torch.nn.utils.rnn import pad_sequence

from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis
from espnet.nets.scorer_interface import ScorerInterface


class BatchHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type."""

    yseq: torch.Tensor                        # Long (batch, seqlen)
    score: torch.Tensor                       # Float (batch,)
    length: List[int]                         # (batch,)
    scores: Dict[str, torch.Tensor] = dict()  # Float (batch,)
    states: Dict[str, Dict] = dict()
    eos: int = -1

    def size(self):
        return self.yseq.size(0)


class BatchBeamSearch(BeamSearch):
    """Batch beam search implementation."""

    def batchfy(self, hs: List[Hypothesis]) -> BatchHypothesis:
        """Convert list to batch."""
        assert len(hs) > 0
        for h in hs:
            assert h.scores.keys() == self.scorers.keys()

        scores = dict()
        states = dict()
        for k, v in self.scorers.items():
            scores[k] = torch.tensor([h.scores[k] for h in hs])
            states[k] = v.merge_states([h.states[k] for h in hs])

        return BatchHypothesis(
            yseq=pad_sequence([h.yseq for h in hs], batch_first=True, padding_value=self.eos),
            length=[len(h.yseq) for h in hs],
            score=torch.tensor([h.score for h in hs]),
            scores=scores,
            states=states
        )

    def unbatchfy(self, batch: BatchHypothesis) -> List[Hypothesis]:
        """Revert batch to list."""
        ret = []
        for i in range(batch.size()):
            ret.append(
                Hypothesis(
                    yseq=batch.yseq[i][:batch.length[i]],
                    score=batch.score[i],
                    scores={k: batch.scores[k][i] for k in self.scorers},
                    states={k: v.select_state(batch.states[k], i) for k, v in self.scorers.items()}
                ))
        return ret

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
                    yseq=self.append_token(hyp.yseq, j)
                    # TODO(karita): implement merge op
                    # scores=self.merge_scores(
                    #     hyp, scores, j, part_scores, part_j),
                    # states=self.merge_states(states, part_states, part_j)
                ))

        # sort and prune 2 x beam -> beam
        return self.top_beam_hyps(best)
