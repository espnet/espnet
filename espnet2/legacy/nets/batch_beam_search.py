"""Parallel beam search module."""

import inspect
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from espnet2.legacy.nets.beam_search import BeamSearch, Hypothesis
from espnet2.legacy.nets.e2e_asr_common import end_detect
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask

logger = logging.getLogger(__name__)

# Score given to slots that must never be selected by `topk`. A large finite
# value is used instead of -inf so that accumulating it over many decoding
# steps can never produce NaN, matching the `logzero` convention of
# `CTCPrefixScoreTH`.
LOG_ZERO = -1.0e10


def _log_zero(dtype: torch.dtype) -> float:
    """Return a maskable score that stays finite in ``dtype``."""
    return max(LOG_ZERO, torch.finfo(dtype).min / 4)


class BatchHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type.

    The hypotheses of ``n_utt`` utterances are held in a single flat batch
    laid out as ``utterance-major``, i.e. the hypotheses of utterance ``b``
    occupy ``b * n_hyp_per_utt`` .. ``(b + 1) * n_hyp_per_utt``. This is the
    layout :class:`CTCPrefixScoreTH` assumes, which recovers the utterance
    index of a hypothesis as ``flat_index // n_hyp_per_utt``.

    When a single utterance is decoded (``n_utt == 1``) the batch is compacted
    after every step, so it holds only the hypotheses that are still running.
    That is not possible with several utterances, because `CTCPrefixScoreTH`
    needs the same number of hypotheses for each of them; there, finished
    hypotheses are flagged in `active` instead and stay in the batch.
    """

    yseq: torch.Tensor = torch.tensor([])  # (batch, maxlen)
    score: torch.Tensor = torch.tensor([])  # (batch,)
    length: torch.Tensor = torch.tensor([])  # (batch,)
    scores: Dict[str, torch.Tensor] = dict()  # values: (batch,)
    states: Dict[str, Dict] = dict()
    hs: List[torch.Tensor] = []  # (batch, maxlen, adim)
    # Number of utterances sharing this batch. This cannot be recovered from
    # the tensors: `len(self)` is `n_utt * n_hyp_per_utt`, and the second
    # factor is only a constant (`beam_size`) once several utterances are
    # batched -- for a single utterance the batch is compacted, so it shrinks
    # as hypotheses end. A batch of 10 slots is therefore ambiguous between
    # one utterance with 10 running hypotheses and 10 utterances with one slot
    # each, and only `search` and `post_process` need the distinction, to fold
    # the flat batch back into a per-utterance view. `n_utt` is stored rather
    # than `n_hyp_per_utt` because it stays fixed for a whole decode.
    # The default is what every construction site that predates utterance
    # batching means, so those keep working untouched.
    n_utt: int = 1
    # Which slots still hold a hypothesis to expand. None means all of them,
    # which is again what pre-existing callers mean.
    active: Optional[torch.Tensor] = None  # (batch,), bool
    # Which utterances have finished decoding. None means none of them.
    done: Optional[torch.Tensor] = None  # (n_utt,), bool

    def __len__(self) -> int:
        """Return a batch size."""
        return len(self.length)

    @property
    def n_hyp_per_utt(self) -> int:
        """Return the number of slots each utterance owns."""
        return len(self) // self.n_utt

    def active_mask(self) -> torch.Tensor:
        """Return the `active` flags, materializing the all-active default."""
        if self.active is not None:
            return self.active
        return torch.ones(len(self), dtype=torch.bool, device=self.yseq.device)

    def done_mask(self) -> torch.Tensor:
        """Return the `done` flags, materializing the none-done default."""
        if self.done is not None:
            return self.done
        return torch.zeros(self.n_utt, dtype=torch.bool, device=self.yseq.device)

    def replace(self, **kwargs) -> "BatchHypothesis":
        """Return a copy with some fields replaced.

        NOTE: ``NamedTuple._replace`` cannot be used because this class
        overrides ``__len__``, which ``_replace`` relies on internally.
        """
        fields = {k: getattr(self, k) for k in BatchHypothesis._fields}
        fields.update(kwargs)
        return BatchHypothesis(**fields)


class BatchBeamSearch(BeamSearch):
    """Batch beam search implementation.

    Hypotheses are vectorized over the beam, and optionally over utterances as
    well: pass a batched encoder output ``(n_utt, T, D)`` to :meth:`forward`
    and a whole minibatch is decoded with a single set of scorer calls, which
    is much faster on an accelerator than decoding one utterance at a time.
    Decoding a single utterance, by passing ``(T, D)``, is the ``n_utt == 1``
    case of the same search and returns a plain n-best list as before.
    """

    def batchfy(self, hyps: List[Hypothesis], n_utt: int = 1) -> BatchHypothesis:
        """Convert list to batch."""
        if len(hyps) == 0:
            return BatchHypothesis()

        if self.return_hs:
            hs = [h.hs for h in hyps]
        else:
            hs = []

        return BatchHypothesis(
            yseq=pad_sequence(
                [h.yseq for h in hyps], batch_first=True, padding_value=self.eos
            ),
            length=torch.tensor([len(h.yseq) for h in hyps], dtype=torch.int64),
            score=torch.tensor([h.score for h in hyps]),
            scores={k: torch.tensor([h.scores[k] for h in hyps]) for k in self.scorers},
            states={k: [h.states[k] for h in hyps] for k in self.scorers},
            hs=hs,
            n_utt=n_utt,
        )

    def _batch_select(self, hyps: BatchHypothesis, ids: List[int]) -> BatchHypothesis:
        if self.return_hs:
            hs = [hyps.hs[i] for i in ids]
        else:
            hs = []

        return BatchHypothesis(
            yseq=hyps.yseq[ids],
            score=hyps.score[ids],
            length=hyps.length[ids],
            scores={k: v[ids] for k, v in hyps.scores.items()},
            states={
                k: [self.scorers[k].select_state(v, i) for i in ids]
                for k, v in hyps.states.items()
            },
            hs=hs,
            n_utt=hyps.n_utt,
            done=hyps.done,
        )

    def _select(self, hyps: BatchHypothesis, i: int) -> Hypothesis:
        return Hypothesis(
            yseq=hyps.yseq[i, : hyps.length[i]],
            score=hyps.score[i],
            scores={k: v[i] for k, v in hyps.scores.items()},
            states={
                k: self.scorers[k].select_state(v, i) for k, v in hyps.states.items()
            },
            hs=hyps.hs[i] if self.return_hs else [],
        )

    def unbatchfy(self, batch_hyps: BatchHypothesis) -> List[Hypothesis]:
        """Revert batch to list."""
        return [
            Hypothesis(
                yseq=batch_hyps.yseq[i][: batch_hyps.length[i]],
                score=batch_hyps.score[i],
                scores={k: batch_hyps.scores[k][i] for k in self.scorers},
                states={
                    k: v.select_state(batch_hyps.states[k], i)
                    for k, v in self.scorers.items()
                },
                hs=batch_hyps.hs[i] if self.return_hs else [],
            )
            for i in range(len(batch_hyps.length))
        ]

    def batch_beam(
        self, weighted_scores: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-compute topk full token ids and partial token ids.

        This is the single-utterance case of the pruning done in
        :meth:`search`; it is kept as a separate method because subclasses
        override it.

        Args:
            weighted_scores (torch.Tensor): The weighted sum scores for each tokens.
                Its shape is `(n_beam, self.vocab_size)`.
            ids (torch.Tensor): The partial token ids to compute topk.
                Its shape is `(n_beam, self.pre_beam_size)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                The topk full (prev_hyp, new_token) ids
                and partial (prev_hyp, new_token) ids.
                Their shapes are all `(self.beam_size,)`

        """
        top_ids = weighted_scores.view(-1).topk(self.beam_size)[1]
        # Because of the flatten above, `top_ids` is organized as:
        # [hyp1 * V + token1, hyp2 * V + token2, ..., hypK * V + tokenK],
        # where V is `self.n_vocab` and K is `self.beam_size`
        prev_hyp_ids = torch.div(top_ids, self.n_vocab, rounding_mode="trunc")
        new_token_ids = top_ids % self.n_vocab
        return prev_hyp_ids, new_token_ids, prev_hyp_ids, new_token_ids

    # ------------------------------------------------------------------
    # encoder-side padding
    # ------------------------------------------------------------------
    @staticmethod
    def supports_xs_mask(scorer: Any) -> bool:
        """Return whether ``scorer.batch_score`` accepts an ``xs_mask``."""
        try:
            params = inspect.signature(scorer.batch_score).parameters
        except (TypeError, ValueError):
            return False
        return "xs_mask" in params

    def _xs_mask_capable(self, key: str) -> bool:
        """Cached form of :meth:`supports_xs_mask`, keyed by scorer name.

        The cache is built lazily rather than in ``__init__`` so that this
        class stays usable through the in-place ``__class__`` assignment that
        the inference scripts do.
        """
        cache = getattr(self, "_xs_mask_support", None)
        if cache is None or key not in cache:
            cache = {k: self.supports_xs_mask(d) for k, d in self.full_scorers.items()}
            self._xs_mask_support = cache
        return cache[key]

    def _expand_over_beam(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Repeat each utterance `beam_size` times and build its padding mask.

        Returns `(xs, xs_mask)` where `xs` is `(n_utt * beam, T, D)` in the
        utterance-major layout. `xs_mask` is None when no utterance is padded,
        so a batch of equally long utterances takes exactly the same code path
        as a single utterance.
        """
        n_utt, max_len = x.size(0), x.size(1)
        xs = (
            x.unsqueeze(1)
            .repeat(1, self.beam_size, 1, 1)
            .view(n_utt * self.beam_size, max_len, -1)
        )
        if bool((x_lengths < max_len).any()):
            mask = ~make_pad_mask(x_lengths, maxlen=max_len)  # (n_utt, T)
            xs_mask = (
                mask.unsqueeze(1)
                .repeat(1, self.beam_size, 1)
                .view(n_utt * self.beam_size, 1, max_len)
                .to(x.device)
            )
        else:
            xs_mask = None
        return xs, xs_mask

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------
    def init_hyp(
        self, x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None
    ) -> BatchHypothesis:
        """Get an initial hypothesis data.

        Args:
            x (torch.Tensor): The encoder output feature, either `(T, D)` for
                a single utterance or `(n_utt, T, D)` for a batch.
            x_lengths (torch.Tensor): Encoder output lengths `(n_utt,)`,
                needed so that the CTC prefix scores ignore padded frames.

        Returns:
            BatchHypothesis: The initial hypotheses. A single utterance starts
            from one hypothesis, as in `BeamSearch`; a batch starts from a
            full `(n_utt, beam)` grid in which only the first beam of each
            utterance is active, which has the same effect.

        """
        n_utt = 1 if x.dim() == 2 else x.size(0)

        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            if (
                x.dim() == 3
                and "xs_lengths" in inspect.signature(d.batch_init_state).parameters
            ):
                init_states[k] = d.batch_init_state(x, xs_lengths=x_lengths)
            else:
                init_states[k] = d.batch_init_state(x)
            init_scores[k] = 0.0

        primers = self._primers(n_utt, x.device)
        if n_utt == 1:
            return self.batchfy(
                [
                    Hypothesis(
                        score=0.0,
                        scores=init_scores,
                        states=init_states,
                        hs=[],
                        yseq=primers[0],
                    )
                ]
            )

        beam = self.beam_size
        n_bh = n_utt * beam
        # The beams of an utterance are all identical at this point, so only
        # the first one may be expanded; otherwise the first `topk` would
        # return `beam` copies of the same token.
        active = torch.zeros(n_bh, dtype=torch.bool, device=x.device)
        active[::beam] = True
        yseq = torch.stack([p for p in primers for _ in range(beam)])
        return BatchHypothesis(
            yseq=yseq,
            length=torch.full(
                (n_bh,), yseq.size(1), dtype=torch.int64, device=x.device
            ),
            score=torch.zeros(n_bh, dtype=x.dtype, device=x.device),
            scores={
                k: torch.zeros(n_bh, dtype=x.dtype, device=x.device)
                for k in self.scorers
            },
            states={k: [v] * n_bh for k, v in init_states.items()},
            hs=[[] for _ in range(n_bh)] if self.return_hs else [],
            n_utt=n_utt,
            active=active,
        )

    @staticmethod
    def _is_per_utt_primer(primer: Any) -> bool:
        """Return whether `hyp_primer` holds one sequence per utterance."""
        return (
            primer is not None
            and len(primer) > 0
            and isinstance(primer[0], (list, tuple, torch.Tensor))
        )

    def _primers(self, n_utt: int, device: torch.device) -> List[torch.Tensor]:
        """Return one primer token sequence per utterance.

        `hyp_primer` may be a single sequence shared by the whole batch, or a
        list of per-utterance sequences (used by S2T to condition on the
        language/task symbols and on the previous text). Per-utterance primers
        must share a length, because all the hypotheses of a batch are
        advanced in lock step.
        """
        primer = [self.sos] if self.hyp_primer is None else self.hyp_primer

        if self._is_per_utt_primer(primer):
            if len(primer) != n_utt:
                raise ValueError(
                    f"got {len(primer)} hyp primers for {n_utt} utterances"
                )
            primers = [
                torch.as_tensor(p, dtype=torch.int64, device=device) for p in primer
            ]
            lengths = {len(p) for p in primers}
            if len(lengths) != 1:
                raise ValueError(
                    "all hyp primers in a batch must have the same length, "
                    f"got {sorted(lengths)}. Decode these utterances with "
                    "--batch_size 1, or pad the primers to a common length."
                )
            return primers

        single = torch.as_tensor(primer, dtype=torch.int64, device=device)
        return [single] * n_utt

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def score_full(
        self,
        hyp: BatchHypothesis,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
        xs_mask: Optional[torch.Tensor] = None,
        pre_xs_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (BatchHypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature (n_batch, T, D)
            pre_x (torch.Tensor): Encoded speech feature for sequential attn
                Sequential attn computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.
            xs_mask (torch.Tensor): Non-padding mask of `x` (n_batch, 1, T),
                or None when every utterance fills the whole tensor.
            pre_xs_mask (torch.Tensor): Non-padding mask of `pre_x`.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        hs = None
        for k, d in self.full_scorers.items():
            kwargs = dict()
            if "decoder" in k:
                if xs_mask is not None and self._xs_mask_capable(k):
                    kwargs["xs_mask"] = xs_mask
                    if pre_x is not None:
                        kwargs["pre_xs_mask"] = pre_xs_mask
                if self.return_hs:
                    kwargs["return_hs"] = True
            if "decoder" in k and self.return_hs:
                (scores[k], hs), states[k] = d.batch_score(
                    hyp.yseq, hyp.states[k], x, **kwargs
                )
            elif "decoder" in k and pre_x is not None:
                scores[k], states[k] = d.batch_score(
                    hyp.yseq, hyp.states[k], x, pre_x, **kwargs
                )
            else:
                scores[k], states[k] = d.batch_score(
                    hyp.yseq, hyp.states[k], x, **kwargs
                )

        if self.return_hs:
            return hs, scores, states
        return scores, states

    def score_partial(
        self,
        hyp: BatchHypothesis,
        ids: torch.Tensor,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.part_scorers`.

        Args:
            hyp (BatchHypothesis): Hypothesis with prefix tokens to score
            ids (torch.Tensor): 2D tensor of new partial tokens to score
            x (torch.Tensor): Corresponding input feature
            pre_x (torch.Tensor): Encoded speech feature for sequential attn (T, D)
                Sequential attn computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            if "ctc" in k and pre_x is not None:
                scores[k], states[k] = d.batch_score_partial(
                    hyp.yseq, ids, hyp.states[k], pre_x
                )
            else:
                scores[k], states[k] = d.batch_score_partial(
                    hyp.yseq, ids, hyp.states[k], x
                )
        return scores, states

    def merge_states(self, states: Any, part_states: Any, part_idx: int) -> Any:
        """Merge states for new hypothesis.

        Args:
            states: states of `self.full_scorers`
            part_states: states of `self.part_scorers`
            part_idx (int): The new token id for `part_scores`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are states of the scorers.

        """
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, v in part_states.items():
            new_states[k] = v
        return new_states

    # ------------------------------------------------------------------
    # one decoding step
    # ------------------------------------------------------------------
    def search(
        self,
        running_hyps: BatchHypothesis,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
        xs_mask: Optional[torch.Tensor] = None,
        pre_xs_mask: Optional[torch.Tensor] = None,
    ) -> BatchHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature. Either `(T, D)` for a
                single utterance, which is replicated over the hypotheses, or
                already replicated as `(n_batch, T, D)`.
            pre_x (torch.Tensor): Encoded speech feature for sequential attention
            xs_mask (torch.Tensor): Non-padding mask of `x`
            pre_xs_mask (torch.Tensor): Non-padding mask of `pre_x`

        Returns:
            BatchHypothesis: `beam_size` best hypotheses per utterance

        """
        n_batch = len(running_hyps)
        n_utt = running_hyps.n_utt
        part_ids = None  # no pre-beam

        if x.dim() == 2:
            x = x.expand(n_batch, *x.shape)
        if pre_x is not None and pre_x.dim() == 2:
            pre_x = pre_x.expand(n_batch, *pre_x.shape)

        weighted_scores = torch.zeros(
            n_batch, self.n_vocab, dtype=x.dtype, device=x.device
        )
        # NOTE: the mask arguments are only passed when they are actually
        # needed, so that subclasses overriding `score_full` with the older
        # signature keep working for the single-utterance case.
        mask_kwargs = dict()
        if xs_mask is not None:
            mask_kwargs["xs_mask"] = xs_mask
        if pre_xs_mask is not None:
            mask_kwargs["pre_xs_mask"] = pre_xs_mask
        if self.return_hs:
            hs, scores, states = self.score_full(
                running_hyps, x, pre_x=pre_x, **mask_kwargs
            )
        else:
            scores, states = self.score_full(
                running_hyps, x, pre_x=pre_x, **mask_kwargs
            )

        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]
        # partial scoring
        if self.do_pre_beam:
            pre_beam_scores = (
                weighted_scores
                if self.pre_beam_score_key == "full"
                else scores[self.pre_beam_score_key]
            )
            part_ids = torch.topk(pre_beam_scores, self.pre_beam_size, dim=-1)[1]
        # NOTE(takaaki-hori): Unlike BeamSearch, we assume that score_partial returns
        # full-size score matrices, which has non-zero scores for part_ids and zeros
        # for others.
        part_scores, part_states = self.score_partial(running_hyps, part_ids, x, pre_x)
        for k in self.part_scorers:
            weighted_scores += self.weights[k] * part_scores[k]
        # add previous hyp scores
        weighted_scores += running_hyps.score.to(
            dtype=x.dtype, device=x.device
        ).unsqueeze(1)
        if running_hyps.active is not None:
            # never expand a slot whose hypothesis already ended
            weighted_scores = weighted_scores.masked_fill(
                ~running_hyps.active.unsqueeze(1), _log_zero(x.dtype)
            )

        # beam pruning, independently per utterance
        if n_utt == 1:
            prev_hyp_ids, new_token_ids, _, _ = self.batch_beam(
                weighted_scores, part_ids
            )
            top_ids = (prev_hyp_ids * self.n_vocab + new_token_ids).unsqueeze(0)
        else:
            top_ids = weighted_scores.view(n_utt, -1).topk(self.beam_size, dim=-1)[1]
            # `top_ids` is organized as `hyp_index * n_vocab + token_index`
            offsets = (
                torch.arange(n_utt, device=x.device) * running_hyps.n_hyp_per_utt
            ).unsqueeze(1)
            prev_hyp_ids = (
                torch.div(top_ids, self.n_vocab, rounding_mode="trunc") + offsets
            ).view(-1)
            new_token_ids = (top_ids % self.n_vocab).view(-1)

        return self._make_batch(
            # `CTCPrefixScoreTH.index_select_state` can only reorder a state
            # in place, so the fast path needs as many new hypotheses as old
            # ones. That always holds for a batch of utterances; for a single
            # utterance it does not right after some hypotheses ended.
            batch_select=len(prev_hyp_ids) == n_batch,
            running_hyps=running_hyps,
            weighted_scores=weighted_scores,
            scores=scores,
            states=states,
            part_scores=part_scores,
            part_states=part_states,
            prev_hyp_ids=prev_hyp_ids,
            new_token_ids=new_token_ids,
            top_ids=top_ids,
            hs=hs if self.return_hs else None,
        )

    def _make_batch(
        self,
        batch_select: bool,
        running_hyps: BatchHypothesis,
        weighted_scores: torch.Tensor,
        scores: Dict[str, torch.Tensor],
        states: Dict[str, Any],
        part_scores: Dict[str, torch.Tensor],
        part_states: Dict[str, Any],
        prev_hyp_ids: torch.Tensor,
        new_token_ids: torch.Tensor,
        top_ids: torch.Tensor,
        hs: Optional[torch.Tensor],
    ) -> BatchHypothesis:
        """Assemble the next batch of hypotheses from the pruned candidates."""
        n_out = len(prev_hyp_ids)
        # `batchfy` keeps the bookkeeping tensors on the CPU, while the scorer
        # outputs live on the model device; line them up before indexing.
        device = weighted_scores.device

        new_yseq = torch.cat(
            [running_hyps.yseq.to(device)[prev_hyp_ids], new_token_ids.unsqueeze(1)],
            dim=1,
        )
        new_score = weighted_scores[prev_hyp_ids, new_token_ids]

        new_scores = dict()
        for k, v in list(scores.items()) + list(part_scores.items()):
            new_scores[k] = (
                running_hyps.scores[k].to(device)[prev_hyp_ids]
                + v[prev_hyp_ids, new_token_ids]
            )

        new_states = dict()
        for k, v in states.items():
            new_states[k] = self._select_states(
                self.full_scorers[k], v, prev_hyp_ids, None, top_ids, batch_select
            )
        for k, v in part_states.items():
            new_states[k] = self._select_states(
                self.part_scorers[k],
                v,
                prev_hyp_ids,
                new_token_ids,
                top_ids,
                batch_select,
            )

        if self.return_hs:
            new_hs = [
                running_hyps.hs[int(j)] + [hs[int(j)].squeeze(0)] for j in prev_hyp_ids
            ]
        else:
            new_hs = []

        return BatchHypothesis(
            yseq=new_yseq,
            score=new_score,
            length=torch.full(
                (n_out,), new_yseq.size(1), dtype=torch.int64, device=new_yseq.device
            ),
            scores=new_scores,
            states=new_states,
            hs=new_hs,
            n_utt=running_hyps.n_utt,
            done=running_hyps.done,
        )

    def _select_states(
        self,
        scorer: Any,
        state: Any,
        prev_hyp_ids: torch.Tensor,
        new_token_ids: Optional[torch.Tensor],
        top_ids: torch.Tensor,
        batch_select: bool,
    ) -> Any:
        """Reorder a scorer state according to the pruned hypothesis ids.

        Scorers may provide `batch_select_state` to do this without a Python
        loop over the hypotheses; `CTCPrefixScorer` does. Its argument is
        `top_ids` of shape `(n_utt, beam)` holding
        `hyp_index * n_vocab + token_index`, which is the layout
        :meth:`CTCPrefixScoreTH.index_select_state` expects.
        """
        if batch_select and hasattr(scorer, "batch_select_state"):
            return scorer.batch_select_state(state, top_ids)
        if new_token_ids is None:
            return [scorer.select_state(state, int(j)) for j in prev_hyp_ids]
        return [
            scorer.select_state(state, int(j), int(t))
            for j, t in zip(prev_hyp_ids, new_token_ids)
        ]

    # ------------------------------------------------------------------
    # per-iteration bookkeeping
    # ------------------------------------------------------------------
    def post_process(
        self,
        i: int,
        maxlen: Union[int, List[int]],
        minlen: Union[int, List[int]],
        maxlenratio: float,
        running_hyps: BatchHypothesis,
        ended_hyps: Union[List[Hypothesis], List[List[Hypothesis]]],
    ) -> BatchHypothesis:
        """Perform post-processing of beam search iterations.

        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int or list[int]): The maximum length of tokens in beam
                search, per utterance when several are decoded together.
            minlen (int or list[int]): The minimum length of tokens.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (BatchHypothesis): The running hypotheses in beam search.
            ended_hyps (list): The ended hypotheses. A flat list when a single
                utterance is decoded, one list per utterance otherwise.
                Appended to in place.

        Returns:
            BatchHypothesis: The new running hypotheses.

        """
        n_utt = running_hyps.n_utt
        per_utt = running_hyps.n_hyp_per_utt
        maxlens = [maxlen] * n_utt if isinstance(maxlen, int) else maxlen
        minlens = [minlen] * n_utt if isinstance(minlen, int) else minlen
        if n_utt == 1 and (len(ended_hyps) == 0 or not isinstance(ended_hyps[0], list)):
            ended_per_utt = [ended_hyps]
        else:
            ended_per_utt = ended_hyps

        logger.debug(f"the number of running hypothes: {len(running_hyps)}")
        if self.token_list is not None and len(running_hyps) > 0:
            logger.debug(
                "best hypo: "
                + "".join(
                    [
                        self.token_list[x]
                        for x in running_hyps.yseq[0, 1 : running_hyps.length[0]]
                    ]
                )
            )

        active = running_hyps.active_mask().clone()
        done = running_hyps.done_mask().clone()
        yseq_device = running_hyps.yseq.device
        is_eos = (
            running_hyps.yseq[
                torch.arange(len(running_hyps), device=yseq_device),
                running_hyps.length.to(yseq_device) - 1,
            ]
            == self.eos
        ).tolist()

        for b in range(n_utt):
            if bool(done[b]):
                active[b * per_utt : (b + 1) * per_utt] = False
                continue
            # add <eos> in the final loop to avoid that there are no ended hyps
            at_maxlen = i == maxlens[b] - 1
            if at_maxlen:
                logger.info("adding <eos> in the last position in the loop")
            for j in range(b * per_utt, (b + 1) * per_utt):
                if not bool(active[j]):
                    continue
                if at_maxlen:
                    # NOTE: <eos> is appended to every running hypothesis,
                    # including one that has just produced <eos> itself.
                    hyp = self._select(running_hyps, j)
                    hyp = hyp._replace(yseq=self.append_token(hyp.yseq, self.eos))
                    if i >= minlens[b]:
                        ended_per_utt[b].append(hyp)
                    active[j] = False
                elif is_eos[j]:
                    if i >= minlens[b]:
                        ended_per_utt[b].append(self._select(running_hyps, j))
                    active[j] = False

            if at_maxlen or not bool(active[b * per_utt : (b + 1) * per_utt].any()):
                done[b] = True
                active[b * per_utt : (b + 1) * per_utt] = False
            elif maxlenratio == 0.0 and end_detect(
                [h.asdict() for h in ended_per_utt[b]], i
            ):
                logger.info(f"end detected at {i}")
                done[b] = True
                active[b * per_utt : (b + 1) * per_utt] = False

        if n_utt == 1:
            # A single utterance does not need a fixed number of hypotheses,
            # so drop the finished ones instead of carrying them along.
            remained_ids = torch.nonzero(active, as_tuple=False).view(-1).cpu()
            return self._batch_select(running_hyps, remained_ids).replace(done=done)
        return running_hyps.replace(active=active, done=done)

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        pre_x: torch.Tensor = None,
        x_lengths: Optional[torch.Tensor] = None,
        pre_x_lengths: Optional[torch.Tensor] = None,
    ) -> Union[List[Hypothesis], List[List[Hypothesis]]]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature `(T, D)` for a single
                utterance, or `(n_utt, T, D)` to decode a whole minibatch with
                one set of scorer calls.
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
                If maxlenratio<0.0, its absolute value is interpreted
                as a constant max output length.
            minlenratio (float): Input length ratio to obtain min output length.
                If minlenratio<0.0, its absolute value is interpreted
                as a constant min output length.
            pre_x (torch.Tensor): Encoded speech feature for sequential attn
                Sequential attn computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.
            x_lengths (torch.Tensor): Encoder output lengths `(n_utt,)`. Only
                used for a batched `x`; when omitted every utterance is
                assumed to fill the whole tensor.
            pre_x_lengths (torch.Tensor): Lengths of `pre_x`.

        Returns:
            list[Hypothesis]: N-best decoding results for a single utterance,
            or list[list[Hypothesis]] with one n-best list per utterance when
            `x` is batched.

        """
        batched = x.dim() == 3
        if batched:
            n_utt = x.size(0)
        elif x.dim() == 2:
            n_utt = 1
        else:
            raise ValueError(f"unsupported encoder output shape {tuple(x.shape)}")

        device = x.device
        if batched and x_lengths is None:
            x_lengths = torch.full(
                (n_utt,), x.size(1), dtype=torch.int64, device=device
            )
        if batched and pre_x is not None and pre_x_lengths is None:
            pre_x_lengths = torch.full(
                (n_utt,), pre_x.size(1), dtype=torch.int64, device=device
            )

        # set length bounds
        inp = x if pre_x is None else pre_x
        if batched:
            inp_lengths = (x_lengths if pre_x is None else pre_x_lengths).tolist()
        else:
            inp_lengths = [inp.shape[0]]
        maxlens, minlens = [], []
        for n in inp_lengths:
            if maxlenratio == 0:
                maxlens.append(n)
            elif maxlenratio < 0:
                maxlens.append(-1 * int(maxlenratio))
            else:
                maxlens.append(max(1, int(maxlenratio * n)))
            if minlenratio < 0:
                minlens.append(-1 * int(minlenratio))
            else:
                minlens.append(int(minlenratio * n))
        logger.info(f"decoder input length: {inp_lengths}")
        logger.info(f"max output length: {maxlens}")
        logger.info(f"min output length: {minlens}")

        if batched:
            # replicate the encoder output over the beam once, outside the loop
            xs, xs_mask = self._expand_over_beam(x, x_lengths)
            if pre_x is not None:
                pre_xs, pre_xs_mask = self._expand_over_beam(pre_x, pre_x_lengths)
            else:
                pre_xs, pre_xs_mask = None, None
            if xs_mask is not None:
                unmasked = [
                    k
                    for k in self.full_scorers
                    if "decoder" in k and not self._xs_mask_capable(k)
                ]
                if unmasked:
                    logger.warning(
                        f"{unmasked} do not accept an xs_mask in batch_score, so "
                        "the padded encoder frames are attended to. Results may "
                        "differ from decoding one utterance at a time."
                    )
        else:
            xs, xs_mask, pre_xs, pre_xs_mask = x, None, pre_x, None

        # main loop of prefix search
        # NOTE: `init_hyp` and `post_process` are hooks that subclasses
        # override. Call them the way they were called before utterance
        # batching whenever a single utterance is decoded, so that an override
        # written against the old signature keeps working.
        if batched:
            running_hyps = self.init_hyp(
                x if pre_x is None else pre_x,
                x_lengths if pre_x is None else pre_x_lengths,
            )
        else:
            running_hyps = self.init_hyp(x if pre_x is None else pre_x)
        ended_hyps: List[List[Hypothesis]] = [[] for _ in range(n_utt)]
        for i in range(max(maxlens)):
            logger.debug("position " + str(i))
            best = self.search(
                running_hyps,
                xs,
                pre_x=pre_xs,
                xs_mask=xs_mask,
                pre_xs_mask=pre_xs_mask,
            )
            running_hyps = self.post_process(
                i,
                maxlens if batched else maxlens[0],
                minlens if batched else minlens[0],
                maxlenratio,
                best,
                ended_hyps if batched else ended_hyps[0],
            )
            if bool(running_hyps.done_mask().all()):
                logger.info(f"decoding finished at position {i}")
                break

        results = [self._nbest(h) for h in ended_hyps]

        # retry the utterances that produced nothing, as `BeamSearch` does
        retry = [b for b in range(n_utt) if len(results[b]) == 0]
        if retry:
            logger.warning(
                f"there is no N-best results for utterances {retry}, "
                "perform recognition again with smaller minlenratio."
            )
            if minlenratio >= 0.1:
                sub_minlenratio = max(0.0, minlenratio - 0.1)
                if not batched:
                    return self.forward(x, maxlenratio, sub_minlenratio, pre_x)
                idx = torch.as_tensor(retry, dtype=torch.int64, device=device)
                # the recursion decodes a subset of the batch, so a
                # per-utterance primer has to be narrowed down to match
                primer = self.hyp_primer
                try:
                    if self._is_per_utt_primer(primer):
                        self.hyp_primer = [primer[b] for b in retry]
                    sub = self.forward(
                        x.index_select(0, idx),
                        maxlenratio,
                        sub_minlenratio,
                        pre_x.index_select(0, idx) if pre_x is not None else None,
                        x_lengths.index_select(0, idx),
                        (
                            pre_x_lengths.index_select(0, idx)
                            if pre_x is not None
                            else None
                        ),
                    )
                finally:
                    self.hyp_primer = primer
                for n, b in enumerate(retry):
                    results[b] = sub[n]

        for b, nbest in enumerate(results):
            self._log_best(b if batched else None, nbest, maxlens[b])
        return results if batched else results[0]

    def _nbest(self, ended: List[Hypothesis]) -> List[Hypothesis]:
        """Sort the finished hypotheses of one utterance."""
        if self.normalize_length:
            # NOTE (Jinchuan): -1 since hyp starts with <sos> and
            # initially has score of 0.0
            return sorted(
                ended, key=lambda x: x.score / (len(x.yseq) - 1), reverse=True
            )
        return sorted(ended, key=lambda x: x.score, reverse=True)

    def _log_best(self, b: Optional[int], nbest: List[Hypothesis], maxlen: int) -> None:
        """Report the best hypothesis of one utterance."""
        tag = "" if b is None else f"utt {b}: "
        if len(nbest) == 0:
            logger.warning(f"{tag}there is no N-best results")
            return
        best = nbest[0]
        for k, v in best.scores.items():
            logger.info(
                f"{tag}{v:6.2f} * {self.weights[k]:3} = "
                f"{v * self.weights[k]:6.2f} for {k}"
            )
        logger.info(f"{tag}total log probability: {best.score:.2f}")
        logger.info(
            f"{tag}normalized log probability: {best.score / len(best.yseq):.2f}"
        )
        logger.info(f"{tag}total number of ended hypotheses: {len(nbest)}")
        if self.token_list is not None:
            logger.info(
                f"{tag}best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        if best.yseq[1:-1].shape[0] == maxlen:
            logger.warning(
                f"{tag}best hypo length: {best.yseq[1:-1].shape[0]} "
                f"== max output length: {maxlen}"
            )
            logger.warning(
                "decoding may be stopped by the max output length limitation, "
                + "please consider to increase the maxlenratio."
            )
