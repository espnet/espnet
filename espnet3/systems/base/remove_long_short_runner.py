"""Runner for parallel long/short utterance duration filtering."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import soundfile as sf

from espnet3.parallel.base_runner import BaseRunner

logger = logging.getLogger(__name__)


class RemoveLongShortRunner(BaseRunner):
    """Runner for filtering utterances by audio duration in parallel.

    Each shard appends its status dicts to a shard-local ``results.jsonl``
    file, and :meth:`merge` reads every shard file back and re-sorts by
    ``idx``, so callers receive results in manifest order regardless of
    shard completion order.
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

    @staticmethod
    def open_writers(shard_dir: Optional[Path], **env) -> Dict[str, Any]:
        """Open the shard-local JSONL results file."""
        results_path = Path(shard_dir) / "results.jsonl"
        return {"results": results_path.open("w", encoding="utf-8")}

    @staticmethod
    def write_record(
        writers: Dict[str, Any],
        result: Any,
        state: Dict[str, Any],
        **env,
    ) -> None:
        """Append one ``forward`` result (or batch of results) to the shard file."""
        records = result if isinstance(result, list) else [result]
        for record in records:
            writers["results"].write(json.dumps(record) + "\n")

    def merge(self, shard_dirs: List[Path]) -> List[Dict[str, Any]]:
        """Concatenate shard results and restore manifest (``idx``) order.

        Each shard's ``results.jsonl`` holds one JSON object per line, e.g.::

            {"idx": 0, "utt_id": "103_1241_000000_000001", "keep": true}
            {"idx": 1, "utt_id": "103_1241_000000_000002", "keep": false}
            {"idx": 2, "utt_id": "103_1241_000001_000000", "keep": true}
        """
        records: List[Dict[str, Any]] = []
        for shard_dir in shard_dirs:
            results_path = Path(shard_dir) / "results.jsonl"
            if not results_path.exists():
                continue
            with results_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        records.sort(key=lambda r: r["idx"])
        return records
