"""SOT constraint scorer for beam search.

Returns 0.0 for allowed tokens and -inf for forbidden tokens.  Since beam
search sums all scorer outputs, -inf effectively masks forbidden tokens.
"""

from typing import Any, List, Optional, Set, Tuple

import torch

from espnet.nets.scorer_interface import BatchScorerInterface


class SOTConstraintScorer(BatchScorerInterface):
    """SOT constraint scorer that enforces valid SOT output structure.

    Enforced constraints:
      A1 - Always suppress <|notimestamps|>
      A2 - Block-scoped constraints (split by separator)
      A3 - Timestamp pairing (two consecutive ts -> force text;
           single end ts -> force separator/EOS)
      A4 - Non-decreasing timestamps within current block
      A5 - Forced initial timestamp
      A6 - Post-separator: force timestamp + suppress double-separator
    """

    def __init__(
        self,
        vocab_size: int,
        eos: int,
        timestamp_begin: int,
        no_timestamps_token_id: int,
        sot_separator_token_id: Optional[int] = None,
        suppress_token_ids: Optional[List[int]] = None,
        begin_index: int = 1,
    ):
        """Initialize SOTConstraintScorer.

        Args:
            vocab_size: Total vocabulary size.
            eos: End-of-sequence token ID (<|endoftext|>).
            timestamp_begin: Token ID of the first timestamp token (<|0.00|>).
            no_timestamps_token_id: Token ID of <|notimestamps|>.
            sot_separator_token_id: Token ID of <sc> (speaker change separator).
            suppress_token_ids: Token IDs to always suppress (from Whisper
                generation_config.json suppress_tokens list).
            begin_index: Number of primer/prefix tokens in yseq
                         (tokens before this index are not generated).
        """
        self.vocab_size = vocab_size
        self.eos = eos
        self.timestamp_begin = timestamp_begin
        self.no_timestamps_token_id = no_timestamps_token_id
        self.sot_separator_token_id = sot_separator_token_id
        self.begin_index = begin_index
        self._suppress_set: Set[int] = (
            set(suppress_token_ids) if suppress_token_ids else set()
        )
        # Custom special tokens above timestamp_begin (e.g., <sc>)
        self._custom_special_above_ts: Set[int] = set()
        if (
            sot_separator_token_id is not None
            and sot_separator_token_id >= timestamp_begin
        ):
            self._custom_special_above_ts.add(sot_separator_token_id)

    def _force_timestamp(self, scores: torch.Tensor) -> None:
        """Force only timestamp tokens.

        Suppress everything below timestamp_begin, restore EOS,
        and suppress custom special tokens above timestamp_begin.
        """
        scores[: self.timestamp_begin] = float("-inf")
        scores[self.eos] = 0.0
        for tok_id in self._custom_special_above_ts:
            scores[tok_id] = float("-inf")

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score a single hypothesis.

        Args:
            y: 1D int64 prefix tokens (full yseq including primer).
            state: Unused (stateless scorer).
            x: Encoder features (used only for device/dtype).

        Returns:
            (scores, None) where scores has shape (vocab_size,):
            0.0 for allowed tokens, -inf for forbidden tokens.
        """
        scores = torch.zeros(self.vocab_size, device=x.device, dtype=x.dtype)

        # A1: Always suppress <|notimestamps|> and suppress_tokens
        scores[self.no_timestamps_token_id] = float("-inf")
        for tok_id in self._suppress_set:
            scores[tok_id] = float("-inf")

        # Extract generated tokens (after primer)
        seq_len = y.shape[0]
        seq = y[self.begin_index :].tolist()

        # A2: Find last separator -> extract current speaker block
        if self.sot_separator_token_id is not None:
            last_sep_pos = None
            for i in range(len(seq) - 1, -1, -1):
                if seq[i] == self.sot_separator_token_id:
                    last_sep_pos = i
                    break

            if last_sep_pos is not None:
                current_block = seq[last_sep_pos + 1 :]
                just_after_separator = len(current_block) == 0
            else:
                current_block = seq
                just_after_separator = False
        else:
            current_block = seq
            just_after_separator = False

        # A6: Post-separator forcing -> force timestamp
        if just_after_separator:
            self._force_timestamp(scores)
            # Suppress separator (no double-separator)
            if self.sot_separator_token_id is not None:
                scores[self.sot_separator_token_id] = float("-inf")
            return scores, None

        # A3: Timestamp pairing scoped to current_block
        last_was_timestamp = (
            len(current_block) >= 1 and current_block[-1] >= self.timestamp_begin
        )
        penultimate_was_timestamp = (
            len(current_block) < 2 or current_block[-2] >= self.timestamp_begin
        )

        if last_was_timestamp:
            if penultimate_was_timestamp:
                # Two consecutive timestamps (or single at block start):
                # text must follow, suppress all timestamps + separator
                scores[self.timestamp_begin :] = float("-inf")
                if self.sot_separator_token_id is not None:
                    scores[self.sot_separator_token_id] = float("-inf")
            else:
                # Single end timestamp after text:
                # suppress text tokens, allow separator/EOS
                scores[: self.eos] = float("-inf")
                if self.sot_separator_token_id is not None:
                    scores[self.sot_separator_token_id] = 0.0

        # A4: Non-decreasing timestamps within current block
        timestamps = [t for t in current_block if t >= self.timestamp_begin]
        if timestamps:
            if last_was_timestamp and not penultimate_was_timestamp:
                timestamp_last = timestamps[-1]
            else:
                timestamp_last = timestamps[-1] + 1
            scores[self.timestamp_begin : timestamp_last] = float("-inf")

        # A5: Forced initial timestamp
        if seq_len == self.begin_index:
            self._force_timestamp(scores)

        return scores, None

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score batch of hypotheses.

        Per-sample loop since constraints depend on per-sample state.
        """
        batch_scores = []
        for i in range(ys.shape[0]):
            score, _ = self.score(ys[i], None, xs[i] if xs.dim() > 1 else xs)
            batch_scores.append(score)
        return torch.stack(batch_scores, dim=0), [None] * ys.shape[0]
