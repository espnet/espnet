"""Whisper timestamp rules for beam search.

The rules reimplement ``whisper.decoding.ApplyTimestampRules`` from OpenAI
Whisper (https://github.com/openai/whisper, MIT licence) on ESPnet's beam
search interfaces, extended with the Serialized Output Training
speaker-change rules.
"""

from typing import Any, List, Optional, Tuple

import torch

from espnet2.legacy.nets.scorer_interface import (
    BatchScorerInterface,
    ScorerInterface,
)


class WhisperTimestampFilter(BatchScorerInterface):
    """Permit only correct Whisper timestamp sequences in beam search.

    The scorer gives each token a score of ``0.0`` or ``-inf``. A score of
    ``0.0`` permits the token. A score of ``-inf`` blocks it. Register the
    scorer with a weight of ``1.0``. The scores are then a hard rule.

    The scorer uses the same rules as
    ``whisper.decoding.ApplyTimestampRules``:

    * The decoder must not write ``<|notimestamps|>``.
    * The first new token must be a timestamp. Use
      ``max_initial_timestamp_index`` to set a limit on this timestamp.
    * Timestamps come in pairs. After one timestamp, the decoder must write a
      second timestamp or the end token. After two timestamps, the decoder
      must write text.
    * Timestamps must not decrease.

    Set ``speaker_change`` to decode Serialized Output Training (SOT) data.
    The scorer then applies the pair rule and the order rule to the current
    speaker block only. A speaker block holds the tokens after the last
    speaker-change token. A new speaker can therefore start again near the
    start of the window. The speaker-change token is permitted only right
    after the timestamp that closes a segment, before the next segment opens.
    A timestamp or the end token must come after it. It must not occur two
    times in sequence.

    If you do not set ``speaker_change``, the rules apply to the full
    sequence. The rules then block each timestamp that is lower than the last
    timestamp of the previous speaker. SOT decoding is not possible.

    Give ``speaker_change`` as a token id, not as a symbol. Each separator
    with one token is usable. Two examples are ``<sc>`` and ``????``. In the
    multilingual Whisper vocabulary, ``????`` is one BPE token.

    Note:
        The scorer needs the Whisper token layout. The text tokens are below
        ``eos``. The timestamps make one continuous range from ``first_time``
        to ``last_time``, above ``eos``. ``__init__`` rejects a layout that
        breaks the second part, because the rule "a segment is closed, so no
        text" is the slice ``mask[:eos]`` and would otherwise blank the
        timestamps with no error raised.

        Find the ids with a symbol lookup. ``OpenAIWhisperTokenIDConverter``
        adds more special tokens. After this, do not read ``timestamp_begin``
        from ``whisper.tokenizer.Tokenizer``. That value is then wrong.

        The scorer does not include the last Whisper rule. That rule compares
        the total probability of the timestamps with the best text token. The
        rule needs the model probabilities, but a scorer receives only the
        tokens. ``WhisperTimestampDecoder`` adds the rule.

    """

    def __init__(
        self,
        first_time: int,
        last_time: int,
        eos: int,
        vocab_size: int,
        sample_begin: int,
        notimestamps: Optional[int] = None,
        speaker_change: Optional[int] = None,
        max_initial_timestamp_index: Optional[int] = None,
    ):
        """Initialize the filter.

        Args:
            first_time (int): Id of the first timestamp token.
            last_time (int): Id of the last timestamp token. This id is
                part of the range.
            eos (int): Id of the end token.
            vocab_size (int): Number of tokens in the vocabulary.
            sample_begin (int): Number of tokens in the decoder prompt. The
                rules do not apply to the tokens before this position.
            notimestamps (Optional[int]): Id of ``<|notimestamps|>``. If you
                give this id, the scorer always blocks the token.
            speaker_change (Optional[int]): Id of the speaker-change token.
                Use ``None`` for the Whisper rules without SOT.
            max_initial_timestamp_index (Optional[int]): Highest permitted
                offset from ``first_time`` for the first timestamp.

        """
        if not first_time <= last_time < vocab_size:
            raise ValueError(
                f"invalid timestamp range [{first_time}, {last_time}] "
                f"for vocab_size {vocab_size}"
            )
        if sample_begin < 0:
            raise ValueError(f"sample_begin must be non-negative: {sample_begin}")
        if speaker_change is not None and first_time <= speaker_change <= last_time:
            raise ValueError(
                f"speaker_change ({speaker_change}) must not be a timestamp token"
            )
        if not eos < first_time:
            # The rules block text with the slice mask[:eos], which is the set
            # of text tokens only when the timestamps sit above eos. Reject
            # any other layout here rather than let that slice silently blank
            # the timestamps at decode time.
            raise ValueError(
                f"eos ({eos}) must be below the timestamp range, which starts "
                f"at first_time ({first_time}); the rules assume the Whisper "
                "layout, with the text tokens below eos and the timestamps "
                "above it"
            )
        if max_initial_timestamp_index is not None and max_initial_timestamp_index < 0:
            raise ValueError(
                "max_initial_timestamp_index must be non-negative: "
                f"{max_initial_timestamp_index}"
            )
        self.first_time = first_time
        self.last_time = last_time
        self.eos = eos
        self.vocab_size = vocab_size
        self.sample_begin = sample_begin
        self.notimestamps = notimestamps
        self.speaker_change = speaker_change
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def set_sample_begin(self, sample_begin: int) -> None:
        """Set the number of tokens in the decoder prompt.

        The decoding options change the length of the prompt. Two examples are
        the timestamp option and the previous-text option. The caller must
        therefore set this value again after each change of the beam search
        primer. Beam search moves all hypotheses of a batch together. One
        value is thus correct for the full batch.

        Args:
            sample_begin (int): Number of tokens in the prompt.

        """
        if sample_begin < 0:
            raise ValueError(f"sample_begin must be non-negative: {sample_begin}")
        self.sample_begin = sample_begin

    def _is_time(self, token: int) -> bool:
        return self.first_time <= token <= self.last_time

    def _mask(self, y: torch.Tensor, dtype, device) -> torch.Tensor:
        neg = -float("inf")
        mask = torch.zeros(self.vocab_size, dtype=dtype, device=device)
        if self.notimestamps is not None:
            mask[self.notimestamps] = neg

        sampled = y[self.sample_begin :].tolist()
        sep = self.speaker_change

        # A block must start with a timestamp. No other token is permitted
        # at the first position.
        if len(sampled) == 0:
            mask[: self.first_time] = neg
            mask[self.last_time + 1 :] = neg
            if self.max_initial_timestamp_index is not None:
                last_allowed = self.first_time + self.max_initial_timestamp_index
                mask[last_allowed + 1 :] = neg
            return mask

        # The same rule applies after a speaker change. The end token is
        # also permitted there.
        if sep is not None and sampled[-1] == sep:
            mask[: self.first_time] = neg
            mask[self.last_time + 1 :] = neg
            mask[self.eos] = 0.0
            mask[sep] = neg
            return mask

        # Use only the current block for the pair rule and the order rule.
        block = sampled
        if sep is not None and sep in sampled:
            last_sep = len(sampled) - 1 - sampled[::-1].index(sep)
            block = sampled[last_sep + 1 :]

        last_is_time = self._is_time(block[-1])
        penult_is_time = len(block) < 2 or self._is_time(block[-2])

        if last_is_time:
            if penult_is_time:
                # Two timestamps are present. Text must come next.
                mask[self.first_time : self.last_time + 1] = neg
            else:
                # A segment is closed. Text must not come next. Everything
                # below eos is text under the layout __init__ enforces, so
                # this leaves eos, the specials and the timestamps open.
                mask[: self.eos] = neg

        # In one block, timestamps must not decrease.
        times = [t for t in block if self._is_time(t)]
        if times:
            mask[self.first_time : times[-1]] = neg

        if sep is not None:
            if last_is_time and not penult_is_time:
                # A speaker can change only at the end of a segment.
                mask[sep] = 0.0
            elif last_is_time and penult_is_time:
                mask[sep] = neg

        return mask

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens.
            x (torch.Tensor): Encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of an additive mask of shape
                ``(vocab_size,)`` and ``None``.

        """
        dtype = x.dtype if x.is_floating_point() else torch.float32
        return self._mask(y, dtype, x.device), None

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor): Encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of batchfied masks with shape
                ``(n_batch, vocab_size)`` and ``None``.

        """
        dtype = xs.dtype if xs.is_floating_point() else torch.float32
        masks = [self._mask(ys[i], dtype, xs.device) for i in range(ys.shape[0])]
        return torch.stack(masks), None


class WhisperTimestampDecoder(ScorerInterface, torch.nn.Module):
    """Add the last Whisper timestamp rule to a decoder scorer.

    ``WhisperTimestampFilter`` holds each rule that needs only the tokens. The
    last Whisper rule needs more. The rule adds the probabilities of all
    timestamps. It compares this total with the best single text token. If the
    total is larger, the rule makes a timestamp necessary:

    .. code-block:: python

        if logprobs[timestamp_begin:].logsumexp(-1) > logprobs[:timestamp_begin].max():
            logits[:timestamp_begin] = -inf

    The rule needs the model probabilities. This class therefore holds the
    decoder scorer and does the same steps as Whisper. First it scores the
    tokens. Then it applies the mask from the filter. Then it makes the
    probabilities sum to one again. Then it applies the last rule.

    Whisper compares the probabilities after the mask. The new sum is thus
    necessary. The Whisper decoder of ESPnet gives ``log_softmax(logits)``.
    This result and the raw logits are different by one constant value.
    ``log_softmax`` gives the same result after a constant shift. The new sum
    of the masked values is therefore equal to the Whisper values.

    Register this class in place of the decoder scorer. Do not also register
    the filter, because this class applies the filter.

    Note:
        The rule blocks each token that is not a timestamp. It also blocks
        ``eos``. Whisper does the same. If you remove the rule, the number of
        timestamps changes. The position of the end of the output can also
        change.

    """

    def __init__(
        self,
        decoder: ScorerInterface,
        timestamp_filter: WhisperTimestampFilter,
    ):
        """Initialize.

        Args:
            decoder (ScorerInterface): Scorer producing log probabilities,
                typically the model decoder.
            timestamp_filter (WhisperTimestampFilter): Filter supplying the
                history based rules.

        """
        super().__init__()
        self.decoder = decoder
        self.filter = timestamp_filter

    # -- state handling is delegated to the wrapped decoder -----------------

    def init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state by delegating to the decoder."""
        return self.decoder.init_state(x)

    def select_state(self, state: Any, i: int, new_id: int = None) -> Any:
        """Select a state by delegating to the decoder."""
        return self.decoder.select_state(state, i, new_id)

    def final_score(self, state: Any) -> float:
        """Score the end of sequence by delegating to the decoder."""
        return self.decoder.final_score(state)

    # -- scoring ------------------------------------------------------------

    def _constrain(self, logp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply the mask, make the sum one again, then apply the last rule.

        This method works on one row.
        """
        first, last = self.filter.first_time, self.filter.last_time
        out = torch.log_softmax(
            logp + self.filter._mask(y, logp.dtype, logp.device), dim=-1
        )
        # A token that is not a timestamp is outside the range
        # [first_time, last_time]. The Whisper timestamps are at the end of
        # the Whisper vocabulary. For Whisper, the code below is therefore
        # equal to ``[:timestamp_begin]``, and the second part costs nothing.
        # The code uses both parts of the range. The rule is then also
        # correct for a vocabulary with tokens above the timestamps.
        best_other = out[:first].max() if first > 0 else None
        if last + 1 < out.shape[-1]:
            tail = out[last + 1 :].max()
            best_other = tail if best_other is None else torch.maximum(best_other, tail)
        if best_other is None:
            # The timestamps fill the vocabulary. No other token is
            # available for the comparison, and no token is left to block.
            return out
        if out[first : last + 1].logsumexp(dim=-1) > best_other:
            out = out.clone()
            out[:first] = -float("inf")
            out[last + 1 :] = -float("inf")
        return out

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens.
            x (torch.Tensor): Encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Constrained log probabilities of shape
                ``(vocab_size,)`` and the decoder state.

        """
        logp, state = self.decoder.score(y, state, x)
        return self._constrain(logp, y), state


class BatchWhisperTimestampDecoder(WhisperTimestampDecoder, BatchScorerInterface):
    """``WhisperTimestampDecoder`` for a decoder that scores a batch.

    The wrapper must advertise the same interface as the decoder it holds.
    ``Speech2Text`` chooses between beam search and batch beam search with
    ``isinstance(scorer, BatchScorerInterface)``, so a wrapper that always
    claimed the batch interface would send a decoder without ``batch_score``
    into batch beam search. Build the pair with :func:`wrap_timestamp_decoder`
    rather than choosing the class by hand.
    """

    def batch_init_state(self, x: torch.Tensor) -> Any:
        """Get an initial batched state by delegating to the decoder."""
        return self.decoder.batch_init_state(x)

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor): Encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, List[Any]]: Constrained log probabilities of
                shape ``(n_batch, vocab_size)`` and the decoder states.

        """
        logp, states = self.decoder.batch_score(ys, states, xs)
        rows = [self._constrain(logp[i], ys[i]) for i in range(ys.shape[0])]
        return torch.stack(rows), states


def wrap_timestamp_decoder(
    decoder: ScorerInterface, timestamp_filter: WhisperTimestampFilter
) -> WhisperTimestampDecoder:
    """Wrap ``decoder`` so the adaptive rule applies, keeping its interface.

    Args:
        decoder (ScorerInterface): Scorer producing log probabilities,
            typically the model decoder.
        timestamp_filter (WhisperTimestampFilter): Filter supplying the
            history based rules.

    Returns:
        WhisperTimestampDecoder: A batch scoring wrapper when ``decoder``
            scores batches, a plain one otherwise.

    """
    cls = (
        BatchWhisperTimestampDecoder
        if isinstance(decoder, BatchScorerInterface)
        else WhisperTimestampDecoder
    )
    return cls(decoder, timestamp_filter)
