"""Runner for parallel audio format conversion."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from espnet3.parallel.base_runner import BaseRunner

logger = logging.getLogger(__name__)


class AudioConversionRunner(BaseRunner):
    """Runner for converting source clips to WAV in parallel.

    Conversion dominates ``create_dataset`` on corpora distributed as video:
    MELD ships 13,708 MP4 clips, which takes over an hour one at a time. Each
    clip is independent, so the work shards cleanly.

    Each shard appends its status dicts to a shard-local ``results.jsonl``
    file, and :meth:`merge` reads every shard file back and re-sorts by
    ``idx``, so callers receive results in job order regardless of shard
    completion order.
    """

    @staticmethod
    def forward(
        idx: Union[int, Iterable[int]],
        jobs: List[Tuple[str, str]],
        ffmpeg: str,
        sampling_rate: int,
        channels: int,
        **env,
    ) -> Union[Dict[str, Any], list]:
        """Convert the clip at the given index or batch of indices.

        Args:
            idx: Single index or iterable of indices into ``jobs``.
            jobs: List of ``(source_path, destination_path)`` pairs.
            ffmpeg: Path to the ``ffmpeg`` executable.
            sampling_rate: Target sampling rate in Hz.
            channels: Target channel count.
            **env: Additional environment entries.

        Returns:
            A status dict for an int index, or a list of status dicts for an
            iterable. Each entry is
            ``{"idx": int, "path": str, "converted": bool}``, where
            ``converted`` is ``False`` for a destination that already existed.

        Raises:
            subprocess.CalledProcessError: If ``ffmpeg`` fails on a clip.
        """
        if isinstance(idx, int):
            return AudioConversionRunner._process_one(
                idx, jobs, ffmpeg, sampling_rate, channels
            )
        return [
            AudioConversionRunner._process_one(i, jobs, ffmpeg, sampling_rate, channels)
            for i in idx
        ]

    @staticmethod
    def _process_one(
        idx: int,
        jobs: List[Tuple[str, str]],
        ffmpeg: str,
        sampling_rate: int,
        channels: int,
    ) -> Dict[str, Any]:
        source, destination = jobs[idx]
        target = Path(destination)
        if target.exists():
            return {"idx": idx, "path": destination, "converted": False}

        target.parent.mkdir(parents=True, exist_ok=True)
        # Convert into a part file and rename, so an interrupted run cannot
        # leave a truncated WAV that the next run would accept as done.
        part = target.with_suffix(target.suffix + ".part")
        subprocess.run(
            [
                ffmpeg,
                "-i",
                source,
                "-ac",
                str(channels),
                "-ar",
                str(sampling_rate),
                "-f",
                "wav",
                "-vn",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                str(part),
            ],
            check=True,
        )
        part.replace(target)
        return {"idx": idx, "path": destination, "converted": True}

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
        """Concatenate shard results and restore job (``idx``) order.

        Each shard's ``results.jsonl`` holds one JSON object per line, e.g.:

        .. code-block:: text

            {"idx": 0, "path": "data/wav/train/a.wav", "converted": true}
            {"idx": 1, "path": "data/wav/train/b.wav", "converted": false}
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
