"""Parallel beam search module."""

from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis


class BatchHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type."""

    yseq: torch.Tensor
    score: torch.Tensor
    length: List[int]
    scores: Dict[str, torch.Tensor] = dict()
    states: Dict[str, Dict] = dict()


class BatchBeamSearch(BeamSearch):
    """Batch beam search implementation."""

    def batchfy(self, hyps: List[Hypothesis]) -> BatchHypothesis:
        """Convert list to batch."""
        return BatchHypothesis(
            yseq=pad_sequence([h.yseq for h in hyps], batch_first=True, padding_value=self.eos),
            length=[len(h.yseq) for h in hyps],
            score=torch.tensor([h.score for h in hyps]),
            scores={k: torch.tensor([h.scores[k] for h in hyps]) for k in self.scorers},
            states={k: [h.states[k] for h in hyps] for k in self.scorers}
        )

    def unbatchfy(self, batch_hyps: BatchHypothesis) -> List[Hypothesis]:
        """Revert batch to list."""
        return [
            Hypothesis(
                yseq=batch_hyps.yseq[i][:batch_hyps.length[i]],
                score=batch_hyps.score[i],
                scores={k: batch_hyps.scores[k][i] for k in self.scorers},
                states={k: v.select_state(
                    batch_hyps.states[k], i) for k, v in self.scorers.items()}
            ) for i in range(len(batch_hyps.length))]

    def batch_beam(self, weighted_scores: torch.Tensor, ids: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-compute topk full token ids and partial token ids.

        Args:
            weighted_scores (torch.Tensor): The weighted sum scores for each tokens.
                Its shape is `(n_beam, self.vocab_size)`.
            ids (torch.Tensor): The partial token ids to compute topk.
                Its shape is `(n_beam, self.pre_beam_size)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                The topk full (beam, vocab) ids and partial (beam, vocab) ids.
                Their shapes are all `(self.beam_size,)`

        """
        if not self.do_pre_beam:
            top_ids = weighted_scores.view(-1).topk(self.beam_size)[1]
            beam_ids = top_ids // self.n_vocab
            vocab_ids = top_ids % self.n_vocab
            return beam_ids, vocab_ids, beam_ids, vocab_ids

        raise NotImplementedError("batch decoding with PartialScorer is not supported yet.")

    def search(self, running_hyps: List[Hypothesis], x: torch.Tensor) -> List[Hypothesis]:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            List[Hypotheses]: Best sorted hypotheses

        """
        n_batch = len(running_hyps)
        batch_hyps = self.batchfy(running_hyps)

        # batch scoring
        scores, states = self.score_full(batch_hyps, x.expand(n_batch, *x.shape))
        if self.do_pre_beam:
            part_ids = torch.topk(scores[self.pre_beam_score_key], self.pre_beam_size, dim=-1)[1]
        else:
            part_ids = torch.arange(self.n_vocab, device=x.device).expand(n_batch, self.n_vocab)
        part_scores, part_states = self.score_partial(batch_hyps, part_ids, x)

        # weighted sum scores
        weighted_scores = torch.zeros(n_batch, self.n_vocab, dtype=x.dtype, device=x.device)
        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]
        for k in self.part_scorers:
            weighted_scores[part_ids] += self.weights[k] * part_scores[k]
        weighted_scores += batch_hyps.score.unsqueeze(1)

        # update hyps
        best_hyps = []
        for full_beam, full_vocab, part_beam, part_vocab in zip(*self.batch_beam(weighted_scores, part_ids)):
            prev_hyp = running_hyps[full_beam]
            best_hyps.append(Hypothesis(
                score=weighted_scores[full_beam, full_vocab],
                yseq=self.append_token(prev_hyp.yseq, full_vocab),
                scores=self.merge_scores(
                    prev_hyp.scores,
                    {k: v[full_beam] for k, v in scores.items()}, full_vocab,
                    {k: v[part_beam] for k, v in part_scores.items()}, part_vocab),
                states=self.merge_states(
                    {k: self.full_scorers[k].select_state(v, full_beam) for k, v in states.items()},
                    {k: self.part_scorers[k].select_state(v, part_beam) for k, v in part_states.items()},
                    part_vocab)
            ))
        return best_hyps
