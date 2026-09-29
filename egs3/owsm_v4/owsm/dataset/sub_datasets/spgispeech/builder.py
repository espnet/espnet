"""SPGISpeech as OWSM utterances.

Ported from ``egs2/owsm_v1/s2t1/local/prepare_spgispeech.py``. Each csv row is
one recording with one transcript, so every utterance forms its own group and
``text_prev`` is always ``<na>``.
"""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path
from typing import Iterable, Iterator

from egs3.owsm_v4.owsm.dataset.builder import OWSMBuilder
from egs3.owsm_v4.owsm.dataset.utils import (
    Utterance,
    generate_long_utterances,
    run_parallel,
)
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_BUILDER_CFG = _CONFIG["builder"]
_DATASET_CFG = _CONFIG["dataset"]

SPLITS = tuple(str(split) for split in _CONFIG["splits"])
AUDIO_SUBDIR = str(_BUILDER_CFG["audio_subdir"])
SOURCE_ENV_VAR = str(_BUILDER_CFG["source_env_var"])

PREFIX = str(_DATASET_CFG["prefix"])
LANG = str(_DATASET_CFG["lang"])
TASK = str(_DATASET_CFG["task"])
CACHE_SUBDIR = str(_DATASET_CFG["cache_subdir"])

_AUDIO_CFG = _BUILDER_CFG["audio"]
HEADER_BYTES = int(_AUDIO_CFG["header_bytes"])
BYTES_PER_SECOND = int(_AUDIO_CFG["bytes_per_second"])
READ_HEADERS = bool(_AUDIO_CFG["read_headers"])


def duration_from_filesize(size: int) -> float:
    """Seconds of audio in a ``size``-byte file of this corpus' fixed format."""
    return (size - HEADER_BYTES) / BYTES_PER_SECOND


def read_manifest(source_root: Path, split: str) -> list[tuple[str, int, str]]:
    """Return ``(wav_rel_path, filesize, transcript)`` per csv row, header skipped."""
    csv_path = source_root / f"{split}.csv"
    rows = []
    with csv_path.open(encoding="utf-8") as stream:
        next(stream, None)
        for line in stream:
            line = line.strip()
            if not line:
                continue
            wav_rel_path, size, transcript = line.split("|")
            rows.append((wav_rel_path, int(size), transcript))
    if not rows:
        raise RuntimeError(f"No rows in {csv_path}")
    return rows


def _verify_file(task):
    """Check one file against its manifest row.

    Module-level and argument-only so it can be pickled to a worker.

    Args:
        task: ``(index, path, size, read_header)``.

    Returns:
        ``(index, None)`` when the file matches, otherwise
        ``(index, "<ExcType>: <message>")``.
    """
    index, path, size, read_header = task
    try:
        on_disk = os.stat(path).st_size
        if on_disk != size:
            raise ValueError(f"file is {on_disk} bytes, manifest says {size}")
        if read_header:
            import soundfile as sf

            info = sf.info(str(path))
            from_header = info.frames / info.samplerate
            from_size = duration_from_filesize(size)
            if round(1000 * from_header) != round(1000 * from_size):
                raise ValueError(
                    f"header says {from_header:.3f}s, filesize says {from_size:.3f}s"
                )
    except Exception as exc:  # noqa: BLE001 - recorded per file, not raised
        return index, repr(exc)
    return index, None


class SPGISpeechBuilder(OWSMBuilder):
    """Build SPGISpeech split caches from the raw csv manifests and audio."""

    CORPUS = "spgispeech"
    CACHE_SUBDIR = CACHE_SUBDIR
    SPLITS = SPLITS
    SOURCE_ENV_VAR = SOURCE_ENV_VAR

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Expect a ``<split>.csv`` and a ``spgispeech/<split>/`` tree per split."""
        return all(
            (candidate / f"{split}.csv").is_file()
            and (candidate / AUDIO_SUBDIR / split).is_dir()
            for split in self.SPLITS
        )

    def verify_manifest(
        self,
        audio_dir: Path,
        manifest: list[tuple[str, int, str]],
        read_headers: bool,
    ) -> dict[int, str | None]:
        """Check every file in ``manifest``, returning one entry per row.

        Durations are taken from the manifest rather than the audio, so the
        size on disk has to be confirmed: a truncated or replaced file would
        otherwise produce a plausible but wrong timestamp. ``read_headers``
        additionally opens each file and confirms the sample rate and width the
        duration arithmetic assumes.
        """
        tasks = [
            (index, audio_dir / rel, size, read_headers)
            for index, (rel, size, _) in enumerate(manifest)
        ]
        return dict(run_parallel(_verify_file, tasks))

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        limit: int | None = None,
        read_headers: bool | None = None,
        **_options,
    ) -> Iterator[dict]:
        """Yield cache rows for ``split``, in csv order.

        ``limit`` takes the first N csv rows, so the offline dump comparison
        can run without a built cache.
        """
        manifest = read_manifest(source_root, split)
        if limit is not None:
            manifest = manifest[:limit]

        # Resolved once, not per file: realpath lstats every path component,
        # and doing that per utterance dominates the whole build.
        audio_dir = (source_root / AUDIO_SUBDIR / split).resolve()
        errors = self.verify_manifest(
            audio_dir, manifest, READ_HEADERS if read_headers is None else read_headers
        )

        for index, (rel, size, transcript) in enumerate(manifest):
            stem = rel.removesuffix(".wav").replace("/", "_")
            wav_id = f"{PREFIX}_{split}_{stem}"
            duration = duration_from_filesize(size)
            error = errors.get(index)
            if error is None and (duration <= 0 or not transcript):
                error = f"duration={duration}, text={transcript!r}"
            if error is not None:
                if failures is not None:
                    failures.write(
                        json.dumps({"utt_id": wav_id, "error": error}) + "\n"
                    )
                continue

            utterance = Utterance(
                utt_id=wav_id,
                wav_id=wav_id,
                wav_path=str(audio_dir / rel),
                start_time=0.0,
                end_time=duration,
                lang=f"<{LANG}>",
                task=f"<{TASK}>",
                text=transcript,
                asr_text=transcript,
            )
            for span in generate_long_utterances([utterance]):
                yield {
                    "utt_id": span.utt_id,
                    "wav_path": span.wav_path,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "lang": LANG,
                    "task": TASK,
                    "tgt_lang": "",
                    "text": span.text_with_time,
                    "text_prev": span.prev_text,
                    "text_ctc": span.asr_text,
                }
