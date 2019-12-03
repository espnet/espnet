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

    yseq: torch.Tensor                        # Long (batch, seqlen)
    score: torch.Tensor                       # Float (batch,)
    length: List[int]                         # (batch,)
    scores: Dict[str, torch.Tensor] = dict()  # Float (batch,)
    states: Dict[str, Dict] = dict()


class BatchBeamSearch(BeamSearch):
    """Batch beam search implementation."""

    def batchfy(self, hs: List[Hypothesis]) -> BatchHypothesis:
        """Convert list to batch."""
        assert len(hs) > 0
        for h in hs:
            if h.scores.keys() != self.scorers.keys():
                raise KeyError(f"{h.scores.keys()} != {self.scorers.keys()}")

        scores = dict()
        states = dict()
        for k, v in self.scorers.items():
            scores[k] = torch.tensor([h.scores[k] for h in hs])
            states[k] = [h.states[k] for h in hs]

        return BatchHypothesis(
            yseq=pad_sequence([h.yseq for h in hs],
                              batch_first=True, padding_value=self.eos),
            length=[len(h.yseq) for h in hs],
            score=torch.tensor([h.score for h in hs]),
            scores=scores,
            states=states
        )

    def unbatchfy(self, batch: BatchHypothesis) -> List[Hypothesis]:
        """Revert batch to list."""
        ret = []
        for i in range(len(batch.length)):
            ret.append(
                Hypothesis(
                    yseq=batch.yseq[i][:batch.length[i]],
                    score=batch.score[i],
                    scores={k: batch.scores[k][i] for k in self.scorers},
                    states={k: v.select_state(
                        batch.states[k], i) for k, v in self.scorers.items()}
                ))
        return ret

    def main_beam(self, weighted_scores: torch.Tensor, ids: torch.Tensor) \
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
        # no pre-beam performed
        print(
            f"ids: {ids.shape}, n_vocab: {self.n_vocab}, scores {weighted_scores.size()}")
        if weighted_scores.size(1) == ids.size(1):
            top_ids = weighted_scores.view(-1).topk(self.beam_size)[1]
            beam_ids = top_ids // self.n_vocab
            vocab_ids = top_ids % self.n_vocab
            print(f"beam: {beam_ids}, vocab: {vocab_ids}")
            return beam_ids, vocab_ids, beam_ids, vocab_ids

        raise NotImplementedError
        # # mask pruned in pre-beam not to select in topk
        # masked_scores = torch.empty_like(weighted_scores)
        # masked_scores.fill_(-float("inf"))
        # # TODO(karita): remove this for-loop
        # local_ids = []
        # for batch, i in enumerate(ids):
        #     local_scores = weighted_scores[batch, i]
        #     masked_scores[batch, i] = local_scores
        #     local_ids.append(local_scores.topk())
        # top_ids = masked_scores.view(-1).topk(self.beam_size)[1]
        # local_ids = weighted_scores[ids].topk(self.beam_size)[1]
        # return top_ids, local_ids

    def score(self, running_hyps: List[Hypothesis], x: torch.Tensor) -> List[Hypothesis]:
        """Score running hypotheses for encoded speech x.

        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            List[Hypotheses]: Best sorted hypotheses

        """
        n_batch = len(running_hyps)
        batch_hyps = self.batchfy(running_hyps)
        # TODO(karita): implement batch full-vocab scorer
        scores, states = self.score_full(batch_hyps, x)
        _d = {k: v.shape for k, v in scores.items()}
        print(f"scores: {_d}")

        # TODO(karita): implement batch partial-vocab scorer
        part_ids = self.pre_beam(scores, device=x.device)
        if part_ids.dim() == 1:
            part_ids = part_ids.expand(n_batch, self.n_vocab)
        part_scores, part_states = self.score_partial(batch_hyps, part_ids, x)

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
        # for vocab dim in weighted_scores
        weighted_scores += batch_hyps.score.unsqueeze(1)

        # update hyps
        best = []
        for full_beam, full_vocab, part_beam, part_vocab in \
                zip(*self.main_beam(weighted_scores, part_ids)):
            prev_hyp = running_hyps[full_beam]
            best.append(Hypothesis(
                score=weighted_scores[full_beam, full_vocab],
                yseq=self.append_token(prev_hyp.yseq, full_vocab),
                scores=self.merge_scores(
                    prev_hyp,
                    {k: v[full_beam] for k, v in scores.items()}, full_vocab,
                    {k: v[part_beam] for k, v in part_scores.items()}, part_vocab),
                states=self.merge_states(
                    {k: self.scorers[k].select_state(
                        v, full_beam) for k, v in states.items()},
                    {k: self.scorers[k].select_state(v, part_beam) for k, v in part_states.items()}, part_vocab)
            ))

        # sort and prune 2 x beam -> beam
        return self.top_beam_hyps(best)
