"""Runner for parallel long/short utterance duration filtering."""

import logging
from typing import Any, Dict, Iterable, List, Tuple, Union

import soundfile as sf

from espnet3.parallel.base_runner import BaseRunner

logger = logging.getLogger(__name__)


class RemoveLongShortRunner(BaseRunner):
    """Runner for filtering utterances by audio duration in parallel.

    Each result carries its originating ``idx`` since ``BaseRunner``'s
    parallel dispatch (``parallel_for``) yields results in completion order,
    not submission order - callers must re-sort by ``idx`` before
    reconstructing the manifest.
    """

    @staticmethod
    def forward(
        idx: Union[int, Iterable[int]],
        entries: List[Tuple[str, str, str]],
        min_duration: float,
        max_duration: float,
        **env,
    ) -> Union[Dict[str, Any], list]:
        """Check duration bounds for the given index or batch.

        Args:
            idx: Single index or iterable of indices into ``entries``.
            entries: List of (utt_id, wav_path, line) tuples from the manifest.
            min_duration: Minimum allowed duration in seconds (exclusive).
            max_duration: Maximum allowed duration in seconds (exclusive).
            **env: Additional environment entries.

        Returns:
            A status dict for an int index, or a list of status dicts for an
            iterable. Each entry is
            ``{"idx": int, "utt_id": str, "keep": bool}``.
        """
        if isinstance(idx, int):
            return RemoveLongShortRunner._process_one(
                idx, entries, min_duration, max_duration
            )
        return [
            RemoveLongShortRunner._process_one(i, entries, min_duration, max_duration)
            for i in idx
        ]

    @staticmethod
    def _process_one(
        idx: int,
        entries: List[Tuple[str, str, str]],
        min_duration: float,
        max_duration: float,
    ) -> Dict[str, Any]:
        utt_id, wav_path, _ = entries[idx]
        duration = sf.info(wav_path).duration

        # Strict inequalities to match espnet2 tts.sh awk filter.
        keep = not (duration <= min_duration or duration >= max_duration)
        return {"idx": idx, "utt_id": utt_id, "keep": keep}
