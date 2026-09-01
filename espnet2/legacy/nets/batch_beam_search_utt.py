"""Beam search module batched over utterances as well as hypotheses."""

import inspect
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch

from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
from espnet2.legacy.nets.beam_search import Hypothesis
from espnet2.legacy.nets.e2e_asr_common import end_detect
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask

logger = logging.getLogger(__name__)

# Score assigned to slots that must never be selected by `topk`.
# A large finite value is used instead of -inf so that accumulating it over
# many decoding steps can never produce NaN. This matches the `logzero`
# convention of `CTCPrefixScoreTH`.
LOG_ZERO = -1.0e10


def _log_zero(dtype: torch.dtype) -> float:
    """Return a maskable score that stays finite in ``dtype``."""
    return max(LOG_ZERO, torch.finfo(dtype).min / 4)


class UttBatchHypothesis(NamedTuple):
    """Hypotheses of a whole minibatch of utterances.

    All tensors are laid out as a flattened ``(n_utt, beam)`` grid, i.e. the
    hypothesis of utterance ``b`` and beam ``k`` lives at index ``b * beam + k``.
    This layout is required by :class:`CTCPrefixScoreTH`, which recovers the
    utterance index as ``flat_index // beam``.

    All *active* hypotheses always share the same length, so ``yseq`` needs no
    padding and ``length`` is uniform over active rows.
    """

    yseq: torch.Tensor = torch.tensor([])  # (n_utt * beam, maxlen)
    score: torch.Tensor = torch.tensor([])  # (n_utt * beam,)
    length: torch.Tensor = torch.tensor([])  # (n_utt * beam,)
    scores: Dict[str, torch.Tensor] = dict()  # values: (n_utt * beam,)
    states: Dict[str, Any] = dict()
    hs: List[torch.Tensor] = []  # (n_utt * beam, maxlen, adim)
    # Whether the slot holds a hypothesis that should be expanded further.
    active: torch.Tensor = torch.tensor([])  # (n_utt * beam,), bool

    def __len__(self) -> int:
        """Return the number of slots, i.e. ``n_utt * beam``."""
        return len(self.length)

    def with_active(self, active: torch.Tensor) -> "UttBatchHypothesis":
        """Return a copy with a new ``active`` mask.

        NOTE: ``NamedTuple._replace`` cannot be used because this class
        overrides ``__len__``, which ``_replace`` relies on internally.
        """
        return UttBatchHypothesis(
            yseq=self.yseq,
            score=self.score,
            length=self.length,
            scores=self.scores,
            states=self.states,
            hs=self.hs,
            active=active,
        )


class UttBatchBeamSearch(BatchBeamSearch):
    """Beam search that batches over utterances and hypotheses at once.

    :class:`BatchBeamSearch` vectorizes the ``beam`` axis but decodes one
    utterance at a time, which leaves the accelerator idle for short
    utterances. This class keeps a fixed ``(n_utt, beam)`` grid of hypotheses
    so that a whole minibatch of utterances is decoded in a single set of
    scorer calls.

    Differences from :class:`BatchBeamSearch`:

    * The grid never shrinks. Hypotheses that reached ``<eos>`` are *masked
      out* instead of being removed, because :class:`CTCPrefixScoreTH`
      requires a constant number of hypotheses per utterance.
    * ``maxlen``, ``minlen``, end detection and termination are tracked per
      utterance, so utterances of different lengths can be decoded together.
    * Padded encoder frames are masked out of the decoder cross attention,
      which requires the decoder's ``batch_score`` to accept an ``xs_mask``
      argument (see :meth:`supports_xs_mask`).

    This class adds no instance attributes, so an existing :class:`BeamSearch`
    instance can be converted in place with ``beam_search.__class__ =
    UttBatchBeamSearch``, as is done elsewhere in ESPnet.
    """

    # ------------------------------------------------------------------
    # helpers
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

    def _flat_index(self, n_utt: int, device: torch.device) -> torch.Tensor:
        """Return ``(n_utt, 1)`` offsets that map beam ids to grid ids."""
        return (
            torch.arange(n_utt, device=device).unsqueeze(1) * self.beam_size
        )  # (n_utt, 1)

    def _select(self, hyps: UttBatchHypothesis, i: int) -> Hypothesis:
        """Extract a single hypothesis out of the grid."""
        return Hypothesis(
            yseq=hyps.yseq[i, : hyps.length[i]],
            score=hyps.score[i],
            scores={k: v[i] for k, v in hyps.scores.items()},
            states={
                k: self.scorers[k].select_state(v, i) for k, v in hyps.states.items()
            },
            hs=hyps.hs[i] if self.return_hs else [],
        )

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------
    def init_hyp(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
    ) -> UttBatchHypothesis:
        """Build the initial ``(n_utt, beam)`` hypothesis grid.

        Args:
            x (torch.Tensor): Encoder output of shape ``(n_utt, T, D)``.
            x_lengths (torch.Tensor): Encoder output lengths ``(n_utt,)``.

        Returns:
            UttBatchHypothesis: Grid whose only active slot per utterance is
            beam 0. Keeping the other beams inactive for the first step
            reproduces the single-hypothesis start of :class:`BeamSearch`;
            without it the first ``topk`` would return ``beam`` copies of the
            same token.

        """
        n_utt, beam = x.size(0), self.beam_size
        n_bh = n_utt * beam
        device = x.device

        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            if "xs_lengths" in inspect.signature(d.batch_init_state).parameters:
                init_states[k] = d.batch_init_state(x, xs_lengths=x_lengths)
            else:
                init_states[k] = d.batch_init_state(x)
            init_scores[k] = 0.0

        primers = self._primers(n_utt, device)
        yseq = torch.stack([p for p in primers for _ in range(beam)])  # (n_bh, plen)

        active = torch.zeros(n_bh, dtype=torch.bool, device=device)
        active[::beam] = True

        return UttBatchHypothesis(
            yseq=yseq,
            length=torch.full((n_bh,), yseq.size(1), dtype=torch.int64, device=device),
            score=torch.zeros(n_bh, dtype=x.dtype, device=device),
            scores={
                k: torch.zeros(n_bh, dtype=x.dtype, device=device) for k in self.scorers
            },
            states={k: [v] * n_bh for k, v in init_states.items()},
            hs=[[] for _ in range(n_bh)] if self.return_hs else [],
            active=active,
        )

    def _primers(self, n_utt: int, device: torch.device) -> List[torch.Tensor]:
        """Return one primer token sequence per utterance.

        ``hyp_primer`` may be a single sequence shared by the whole batch, or
        a list of per-utterance sequences (used by S2T to condition on the
        language/task symbols and on the previous text). Per-utterance
        primers must share a length, because all hypotheses of the grid are
        advanced in lock step.
        """
        if self.hyp_primer is None:
            primer = [self.sos]
        else:
            primer = self.hyp_primer

        if len(primer) > 0 and isinstance(primer[0], (list, tuple, torch.Tensor)):
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
        hyp: UttBatchHypothesis,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
        xs_mask: Optional[torch.Tensor] = None,
        pre_xs_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score the whole grid with ``self.full_scorers``.

        Args:
            hyp (UttBatchHypothesis): Grid of running hypotheses.
            x (torch.Tensor): Encoder output ``(n_utt * beam, T, D)``.
            pre_x (torch.Tensor): Encoder output for sequential attention.
            xs_mask (torch.Tensor): Non-padding mask ``(n_utt * beam, 1, T)``
                for ``x``, or None when every utterance has the same length.
            pre_xs_mask (torch.Tensor): Non-padding mask for ``pre_x``.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Scores of shape
            ``(n_utt * beam, n_vocab)`` and the new scorer states.

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

    # ------------------------------------------------------------------
    # one decoding step
    # ------------------------------------------------------------------
    def search(
        self,
        running_hyps: UttBatchHypothesis,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
        xs_mask: Optional[torch.Tensor] = None,
        pre_xs_mask: Optional[torch.Tensor] = None,
    ) -> UttBatchHypothesis:
        """Search the next token for every utterance in the batch.

        Args:
            running_hyps (UttBatchHypothesis): Grid of running hypotheses.
            x (torch.Tensor): Encoder output ``(n_utt * beam, T, D)``.
            pre_x (torch.Tensor): Encoder output for sequential attention.
            xs_mask (torch.Tensor): Non-padding mask for ``x``.
            pre_xs_mask (torch.Tensor): Non-padding mask for ``pre_x``.

        Returns:
            UttBatchHypothesis: A full grid of ``beam`` new hypotheses per
            utterance.

        """
        n_bh = len(running_hyps)
        beam = self.beam_size
        n_utt = n_bh // beam
        part_ids = None  # no pre-beam

        weighted_scores = torch.zeros(
            n_bh, self.n_vocab, dtype=x.dtype, device=x.device
        )
        if self.return_hs:
            hs, scores, states = self.score_full(
                running_hyps, x, pre_x=pre_x, xs_mask=xs_mask, pre_xs_mask=pre_xs_mask
            )
        else:
            scores, states = self.score_full(
                running_hyps, x, pre_x=pre_x, xs_mask=xs_mask, pre_xs_mask=pre_xs_mask
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
        part_scores, part_states = self.score_partial(running_hyps, part_ids, x, pre_x)
        for k in self.part_scorers:
            weighted_scores += self.weights[k] * part_scores[k]

        # add previous hyp scores
        weighted_scores += running_hyps.score.to(
            dtype=x.dtype, device=x.device
        ).unsqueeze(1)
        # never expand a slot whose hypothesis already ended (or a slot that is
        # only a placeholder, as on the first step)
        weighted_scores = weighted_scores.masked_fill(
            ~running_hyps.active.unsqueeze(1), _log_zero(x.dtype)
        )

        # beam pruning, independently per utterance
        top_ids = weighted_scores.view(n_utt, beam * self.n_vocab).topk(beam, dim=-1)[1]
        # `top_ids` is organized as `beam_index * n_vocab + token_index`
        prev_hyp_ids = (
            torch.div(top_ids, self.n_vocab, rounding_mode="trunc")
            + self._flat_index(n_utt, x.device)
        ).view(-1)
        new_token_ids = (top_ids % self.n_vocab).view(-1)

        return self._make_grid(
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

    def _make_grid(
        self,
        running_hyps: UttBatchHypothesis,
        weighted_scores: torch.Tensor,
        scores: Dict[str, torch.Tensor],
        states: Dict[str, Any],
        part_scores: Dict[str, torch.Tensor],
        part_states: Dict[str, Any],
        prev_hyp_ids: torch.Tensor,
        new_token_ids: torch.Tensor,
        top_ids: torch.Tensor,
        hs: Optional[torch.Tensor],
    ) -> UttBatchHypothesis:
        """Assemble the next hypothesis grid from the pruned candidates."""
        n_bh = len(running_hyps)

        new_yseq = torch.cat(
            [running_hyps.yseq[prev_hyp_ids], new_token_ids.unsqueeze(1)], dim=1
        )
        new_score = weighted_scores[prev_hyp_ids, new_token_ids]

        new_scores = dict()
        for k, v in scores.items():
            new_scores[k] = (
                running_hyps.scores[k][prev_hyp_ids] + v[prev_hyp_ids, new_token_ids]
            )
        for k, v in part_scores.items():
            new_scores[k] = (
                running_hyps.scores[k][prev_hyp_ids] + v[prev_hyp_ids, new_token_ids]
            )

        new_states = dict()
        for k, v in states.items():
            new_states[k] = self._select_states(
                self.full_scorers[k], v, prev_hyp_ids, None, top_ids
            )
        for k, v in part_states.items():
            new_states[k] = self._select_states(
                self.part_scorers[k], v, prev_hyp_ids, new_token_ids, top_ids
            )

        if self.return_hs:
            new_hs = [
                running_hyps.hs[int(j)] + [hs[int(j)].squeeze(0)] for j in prev_hyp_ids
            ]
        else:
            new_hs = []

        return UttBatchHypothesis(
            yseq=new_yseq,
            score=new_score,
            length=torch.full(
                (n_bh,),
                new_yseq.size(1),
                dtype=torch.int64,
                device=new_yseq.device,
            ),
            scores=new_scores,
            states=new_states,
            hs=new_hs,
            active=torch.ones(n_bh, dtype=torch.bool, device=new_yseq.device),
        )

    def _select_states(
        self,
        scorer: Any,
        state: Any,
        prev_hyp_ids: torch.Tensor,
        new_token_ids: Optional[torch.Tensor],
        top_ids: torch.Tensor,
    ) -> Any:
        """Reorder a scorer state according to the pruned hypothesis ids.

        Scorers may provide ``batch_select_state`` to do this without a Python
        loop over the ``n_utt * beam`` hypotheses; ``CTCPrefixScorer`` does.
        Its argument is ``top_ids`` of shape ``(n_utt, beam)`` holding
        ``beam_index * n_vocab + token_index``, which is the layout
        :meth:`CTCPrefixScoreTH.index_select_state` expects.
        """
        if hasattr(scorer, "batch_select_state"):
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
        maxlens: List[int],
        minlens: List[int],
        maxlenratio: float,
        running_hyps: UttBatchHypothesis,
        ended_hyps: List[List[Hypothesis]],
        done: List[bool],
    ) -> UttBatchHypothesis:
        """Move finished hypotheses out of the grid.

        Args:
            i (int): Current decoding step.
            maxlens (List[int]): Maximum output length of each utterance.
            minlens (List[int]): Minimum output length of each utterance.
            maxlenratio (float): Input length ratio for the max output length.
            running_hyps (UttBatchHypothesis): Grid produced by :meth:`search`.
            ended_hyps (List[List[Hypothesis]]): Finished hypotheses, per
                utterance. Appended to in place.
            done (List[bool]): Whether each utterance finished decoding.
                Updated in place.

        Returns:
            UttBatchHypothesis: The grid with finished slots deactivated.

        """
        beam = self.beam_size
        n_utt = len(running_hyps) // beam
        active = running_hyps.active.clone()
        is_eos = (running_hyps.yseq[:, -1] == self.eos).tolist()

        if self.token_list is not None:
            best = running_hyps.yseq[0, 1:]
            logger.debug("best hypo: " + "".join([self.token_list[x] for x in best]))

        for b in range(n_utt):
            if done[b]:
                active[b * beam : (b + 1) * beam] = False
                continue
            # add <eos> in the final loop to avoid that there are no ended hyps
            at_maxlen = i == maxlens[b] - 1
            if at_maxlen:
                logger.info(f"utt {b}: adding <eos> in the last position in the loop")
            for k in range(beam):
                j = b * beam + k
                if not bool(active[j]):
                    continue
                if at_maxlen:
                    # NOTE: `BatchBeamSearch` appends <eos> to every running
                    # hypothesis at the last step, including one that just
                    # produced <eos>. That behaviour is kept here so that
                    # results match --batch_size 1 exactly.
                    hyp = self._select(running_hyps, j)
                    hyp = hyp._replace(yseq=self.append_token(hyp.yseq, self.eos))
                    if i >= minlens[b]:
                        ended_hyps[b].append(hyp)
                    active[j] = False
                elif is_eos[j]:
                    if i >= minlens[b]:
                        ended_hyps[b].append(self._select(running_hyps, j))
                    active[j] = False

            if at_maxlen or not bool(active[b * beam : (b + 1) * beam].any()):
                if not at_maxlen:
                    logger.info(f"utt {b}: no hypothesis. Finish decoding.")
                done[b] = True
                active[b * beam : (b + 1) * beam] = False
            elif maxlenratio == 0.0 and end_detect(
                [h.asdict() for h in ended_hyps[b]], i
            ):
                logger.info(f"utt {b}: end detected at {i}")
                done[b] = True
                active[b * beam : (b + 1) * beam] = False

        return running_hyps.with_active(active)

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        pre_x: torch.Tensor = None,
        pre_x_lengths: Optional[torch.Tensor] = None,
    ) -> List[List[Hypothesis]]:
        """Perform beam search on a minibatch of utterances.

        Args:
            x (torch.Tensor): Encoder output ``(n_utt, T, D)``.
            x_lengths (torch.Tensor): Encoder output lengths ``(n_utt,)``.
                When None, every utterance is assumed to occupy all ``T``
                frames, which is the case for fixed-length inputs such as S2T.
            maxlenratio (float): Input length ratio to obtain max output
                length. ``0.0`` (default) uses the end-detect function,
                a negative value is interpreted as a constant max length.
            minlenratio (float): Input length ratio to obtain min output
                length. A negative value is a constant min length.
            pre_x (torch.Tensor): Encoder output for sequential attention.
            pre_x_lengths (torch.Tensor): Lengths of ``pre_x``.

        Returns:
            List[List[Hypothesis]]: N-best hypotheses of each utterance, in
            the order the utterances were given.

        """
        if x.dim() != 3:
            raise ValueError(
                f"expected a batched encoder output (n_utt, T, D), got {tuple(x.shape)}"
            )
        n_utt = x.size(0)
        device = x.device

        if x_lengths is None:
            x_lengths = torch.full(
                (n_utt,), x.size(1), dtype=torch.int64, device=device
            )
        if pre_x is not None and pre_x_lengths is None:
            pre_x_lengths = torch.full(
                (n_utt,), pre_x.size(1), dtype=torch.int64, device=device
            )

        inp_lengths = x_lengths if pre_x is None else pre_x_lengths
        maxlens, minlens = [], []
        for b in range(n_utt):
            n = int(inp_lengths[b])
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
        logger.info(f"decoder input lengths: {inp_lengths.tolist()}")
        logger.info(f"max output lengths: {maxlens}")
        logger.info(f"min output lengths: {minlens}")

        # replicate the encoder output over the beam axis once, outside the loop
        xs, xs_mask = self._expand_over_beam(x, x_lengths)
        if xs_mask is not None:
            unmasked = [
                k
                for k in self.full_scorers
                if "decoder" in k and not self._xs_mask_capable(k)
            ]
            if unmasked:
                logger.warning(
                    f"{unmasked} do not accept an xs_mask in batch_score, so the "
                    "padded encoder frames are attended to. Results may differ "
                    "from --batch_size 1."
                )
        if pre_x is not None:
            pre_xs, pre_xs_mask = self._expand_over_beam(pre_x, pre_x_lengths)
        else:
            pre_xs, pre_xs_mask = None, None

        running_hyps = self.init_hyp(x if pre_x is None else pre_x, inp_lengths)
        ended_hyps: List[List[Hypothesis]] = [[] for _ in range(n_utt)]
        done: List[bool] = [False] * n_utt

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
                i, maxlens, minlens, maxlenratio, best, ended_hyps, done
            )
            if all(done):
                logger.info(f"all utterances finished at position {i}")
                break

        results = [self._nbest(h) for h in ended_hyps]

        # retry the utterances that produced nothing, exactly as `BeamSearch`
        # does, but only for those utterances
        retry = [b for b in range(n_utt) if len(results[b]) == 0]
        if retry:
            logger.warning(
                f"there is no N-best results for utterances {retry}, "
                "perform recognition again with smaller minlenratio."
            )
            if minlenratio >= 0.1:
                idx = torch.as_tensor(retry, dtype=torch.int64, device=device)
                sub = self.forward(
                    x.index_select(0, idx),
                    x_lengths.index_select(0, idx),
                    maxlenratio,
                    max(0.0, minlenratio - 0.1),
                    pre_x.index_select(0, idx) if pre_x is not None else None,
                    (pre_x_lengths.index_select(0, idx) if pre_x is not None else None),
                )
                for n, b in enumerate(retry):
                    results[b] = sub[n]

        for b, nbest in enumerate(results):
            self._log_best(b, nbest, maxlens[b])
        return results

    def _expand_over_beam(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Repeat each utterance ``beam`` times and build its padding mask.

        Returns ``(xs, xs_mask)`` where ``xs`` is ``(n_utt * beam, T, D)`` in
        the ``b * beam + k`` layout. ``xs_mask`` is None when no utterance is
        padded, so that a uniform-length batch takes exactly the same code
        path as ``--batch_size 1``.
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

    def _nbest(self, ended: List[Hypothesis]) -> List[Hypothesis]:
        """Sort the finished hypotheses of one utterance."""
        if self.normalize_length:
            # NOTE: -1 since hyp starts with <sos> and initially scores 0.0
            return sorted(
                ended, key=lambda x: x.score / (len(x.yseq) - 1), reverse=True
            )
        return sorted(ended, key=lambda x: x.score, reverse=True)

    def _log_best(self, b: int, nbest: List[Hypothesis], maxlen: int) -> None:
        """Report the best hypothesis of one utterance."""
        if len(nbest) == 0:
            logger.warning(f"utt {b}: no N-best results")
            return
        best = nbest[0]
        for k, v in best.scores.items():
            logger.info(
                f"utt {b}: {v:6.2f} * {self.weights[k]:3} = "
                f"{v * self.weights[k]:6.2f} for {k}"
            )
        logger.info(f"utt {b}: total log probability: {best.score:.2f}")
        logger.info(
            f"utt {b}: normalized log probability: "
            f"{best.score / len(best.yseq):.2f}"
        )
        logger.info(f"utt {b}: total number of ended hypotheses: {len(nbest)}")
        if self.token_list is not None:
            logger.info(
                f"utt {b}: best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        if best.yseq[1:-1].shape[0] == maxlen:
            logger.warning(
                f"utt {b}: best hypo length: {best.yseq[1:-1].shape[0]} "
                f"== max output length: {maxlen}"
            )
            logger.warning(
                "decoding may be stopped by the max output length limitation, "
                "please consider to increase the maxlenratio."
            )
