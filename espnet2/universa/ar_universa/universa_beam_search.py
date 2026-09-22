"""Metric constraints and pair scheduling for the shared ESPnet beam search."""

from typing import Dict, List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.legacy.nets.beam_search import BeamSearch, Hypothesis
from espnet2.legacy.nets.scorer_interface import (
    BatchScorerInterface,
    PartialScorerInterface,
    ScorerInterface,
)


class MetricConstraintScorer(BatchScorerInterface):
    """Allow unused metric labels and their values with a zero/-inf mask.

    The prefix is the source of truth for metric order and completion; no
    separate per-hypothesis list of unused labels needs to be maintained.
    The scorer also supports the shared BatchBeamSearch scorer interface.
    """

    def __init__(self, vocab_size, labels, beam_masking, use_fixed_order):
        """Store the metric vocabulary, value ranges, and ordering policy."""
        self.vocab_size = vocab_size
        self.labels = labels
        self.beam_masking = beam_masking or {}
        self.use_fixed_order = use_fixed_order

    def score(self, y, state, x):
        """Mask tokens using the alternating label/value prefix after SOS."""
        mask = x.new_full((self.vocab_size,), -float("inf"))
        if len(y) % 2:
            used = set(y[1::2].tolist())
            allowed = [label for label in self.labels if label not in used]
            if self.use_fixed_order:
                allowed = allowed[:1]
            mask[allowed] = 0
        else:
            start, end = self.beam_masking.get(int(y[-1]), (0, self.vocab_size))
            mask[start:end] = 0
        return mask, None

    def batch_score(self, ys, states, xs):
        """Build independent masks for hypotheses and utterances in a batch."""
        return torch.stack([self.score(y, None, x)[0] for y, x in zip(ys, xs)]), [
            None
        ] * len(ys)


class ARUniVERSABeamSearch(BeamSearch):
    """Schedule metric/value pairs using ESPnet's shared token search.

    Each result contains SOS followed by exactly one pair per requested metric.
    EOS is accepted for compatibility with the model's decoder configuration,
    but is neither appended nor scored; completion is determined by pair count.

    Both BeamSearch and BatchBeamSearch normally prune after every token. Here
    all retained label branches compete only after their value is scored. When
    label scores are skipped, every allowed label must reach the value step,
    even with beam_size=1. A mask alone cannot express this pruning schedule.
    Only this pair schedule and fixed-length termination are specialized;
    token expansion, score/state merging, hypotheses, and module registration
    come from BeamSearch. BatchBeamSearch supports batching both hypotheses and
    utterances, but its token-level pruning/termination needs the same schedule
    adaptation before it can replace this single-utterance entry point.
    """

    @typechecked
    def __init__(
        self,
        scorers: Dict[str, Optional[ScorerInterface]],
        weights: Dict[str, float],
        beam_size: int,
        vocab_size: int,
        sos: int,
        eos: int,
        meta_label_for_search: List[int],
        token_list: Optional[List[str]] = None,
        skip_meta_label_score: bool = False,
        beam_masking: Optional[Dict[int, Tuple[int, int]]] = None,
        use_fixed_order: bool = False,
    ):
        """Configure full scorers and the requested metric/value constraints.

        ``beam_masking`` maps metric label IDs to half-open value-token ranges.
        ``skip_meta_label_score`` ignores label scores while still advancing
        scorer states. ``use_fixed_order`` follows ``meta_label_for_search``.
        Other arguments follow :class:`BeamSearch`; partial scorers are not
        supported by this fixed-pair decoder.
        """
        if beam_size < 1:
            raise ValueError("beam_size must be positive")
        if len(set(meta_label_for_search)) != len(meta_label_for_search):
            raise ValueError("Requested metrics must be unique")
        if any(label < 0 or label >= vocab_size for label in meta_label_for_search):
            raise ValueError("Metric label IDs must be within the vocabulary")
        for start, end in (beam_masking or {}).values():
            if not 0 <= start < end <= vocab_size:
                raise ValueError("Beam masking ranges must be within the vocabulary")
        if "metric_constraint" in scorers:
            raise ValueError("metric_constraint is reserved for the constraint scorer")
        if any(
            isinstance(scorer, PartialScorerInterface) and weights.get(name, 0) != 0
            for name, scorer in scorers.items()
        ):
            raise ValueError("Metric pair search requires full ScorerInterface scorers")
        constraint = MetricConstraintScorer(
            vocab_size, meta_label_for_search, beam_masking, use_fixed_order
        )
        super().__init__(
            scorers={**scorers, "metric_constraint": constraint},
            weights={**weights, "metric_constraint": 1.0},
            beam_size=beam_size,
            vocab_size=vocab_size,
            sos=sos,
            eos=eos,
            token_list=token_list,
        )
        self.meta_label_for_search = meta_label_for_search
        self.skip_meta_label_score = skip_meta_label_score

    def score_full(self, hyp, x, pre_x=None):
        """Advance all states, optionally ignoring the model's label scores."""
        scores, states = super().score_full(hyp, x, pre_x)
        if self.skip_meta_label_score and len(hyp.yseq) % 2:
            scores = {
                name: score if name == "metric_constraint" else torch.zeros_like(score)
                for name, score in scores.items()
            }
        return scores, states

    def beam(self, weighted_scores, ids):
        """Keep only finite candidates, including when the beam exceeds a range."""
        allowed = ids[torch.isfinite(weighted_scores[ids])]
        local = weighted_scores[allowed].topk(min(self.beam_size, len(allowed))).indices
        chosen = allowed[local]
        # No pre-beam/partial scoring: full and partial IDs coincide.
        return chosen, chosen

    def search(self, running_hyps, x, pre_x=None):
        """Expand labels per parent, then prune globally after the value step."""
        pair_beam_size = self.beam_size
        try:
            if self.skip_meta_label_score:
                self.beam_size = max(pair_beam_size, len(self.meta_label_for_search))
            labels = []
            for hyp in running_hyps:
                labels.extend(super().search([hyp], x, pre_x))
        finally:
            self.beam_size = pair_beam_size
        return super().search(labels, x, pre_x)

    def forward(self, x: torch.Tensor) -> List[Hypothesis]:
        """Decode a single encoded utterance (T, D) into complete metric pairs."""
        running_hyps = self.init_hyp(x)
        for _ in self.meta_label_for_search:
            running_hyps = self.search(running_hyps, x)
        return running_hyps
