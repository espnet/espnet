"""MuST-C as OWSM utterances.

Ported from ``egs2/owsm_v1/s2t1/local/prepare_must-c.py``. Each segment yields
two utterances in two separate groups: an ASR one whose text is the English
source, and an ST one whose text is the translation and whose CTC text is still
the English source. The groups are separate so that a 30 s span never mixes the
two tasks.
"""

from __future__ import annotations

import json
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Iterable, Iterator

import yaml

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
SOURCE_ENV_VAR = str(_BUILDER_CFG["source_env_var"])
LANGUAGES = tuple(str(lang) for lang in _BUILDER_CFG["languages"])
EXPECTED_FS = int(_BUILDER_CFG["expected_fs"])

PREFIX = str(_DATASET_CFG["prefix"])
LANG = str(_DATASET_CFG["lang"])
CACHE_SUBDIR = str(_DATASET_CFG["cache_subdir"])

try:  # 10x faster on train's multi-hundred-MB yaml
    _YAML_LOADER = yaml.CSafeLoader
except AttributeError:
    _YAML_LOADER = yaml.SafeLoader


def _pair_dir(source_root: Path, lang: str, split: str) -> Path:
    return source_root / f"en-{lang}" / "data" / split


def read_segments(source_root: Path, lang: str, split: str) -> list[dict]:
    """Return ``{wav, offset, duration, src, tgt}`` per segment, in file order."""
    txt_dir = _pair_dir(source_root, lang, split) / "txt"
    with (txt_dir / f"{split}.yaml").open(encoding="utf-8") as stream:
        entries = yaml.load(stream, Loader=_YAML_LOADER)
    with (txt_dir / f"{split}.en").open(encoding="utf-8") as stream:
        source = [line.strip() for line in stream]
    with (txt_dir / f"{split}.{lang}").open(encoding="utf-8") as stream:
        target = [line.strip() for line in stream]

    if not len(entries) == len(source) == len(target):
        raise RuntimeError(
            f"en-{lang}/{split}: {len(entries)} yaml entries, {len(source)} "
            f"source lines, {len(target)} target lines"
        )
    return [
        {
            "wav": entry["wav"],
            "offset": float(entry["offset"]),
            "duration": float(entry["duration"]),
            "src": " ".join(src.split()),
            "tgt": " ".join(tgt.split()),
        }
        for entry, src, tgt in zip(entries, source, target)
    ]


def _verify_wav(task):
    """Check that one talk's audio exists and covers its last segment.

    Module-level and argument-only so it can be pickled to a worker.

    Args:
        task: ``(wav_path, last_end_time, expected_fs)``.

    Returns:
        ``(wav_path, None)`` when usable, otherwise ``(wav_path, error)``.
    """
    import soundfile as sf

    wav_path, last_end, expected_fs = task
    try:
        info = sf.info(str(wav_path))
        if info.samplerate != expected_fs:
            raise ValueError(
                f"sample rate is {info.samplerate}, expected {expected_fs}"
            )
        duration = info.frames / info.samplerate
        if duration + 0.05 < last_end:
            raise ValueError(
                f"audio is {duration:.2f}s but segments run to {last_end:.2f}s"
            )
    except Exception as exc:  # noqa: BLE001 - recorded per talk, not raised
        return str(wav_path), repr(exc)
    return str(wav_path), None


class MuSTCBuilder(OWSMBuilder):
    """Build MuST-C split caches from the released yaml and text files."""

    CORPUS = "must_c"
    CACHE_SUBDIR = CACHE_SUBDIR
    SPLITS = SPLITS
    SOURCE_ENV_VAR = SOURCE_ENV_VAR

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Expect ``en-<lang>/data/<split>/txt/<split>.yaml`` for every pair."""
        return all(
            (_pair_dir(candidate, lang, split) / "txt" / f"{split}.yaml").is_file()
            for lang in LANGUAGES
            for split in self.SPLITS
        )

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        limit: int | None = None,
        languages: Iterable[str] | None = None,
        **_options,
    ) -> Iterator[dict]:
        """Yield cache rows for ``split``: every ST row, each ASR row once.

        The English side is regenerated for every language pair, and upstream
        collapses the copies by utterance id afterwards. Doing it here instead
        keeps the mixture's task ratio right -- without it the ASR side is more
        than four times too large.

        The surviving copy is the first in sorted language order. The pairs
        segment the same talk independently, so two copies sharing an utterance
        id can still differ inside; upstream's winner falls out of directory
        iteration order and is not reproducible, so a deterministic rule is
        chosen instead.
        """
        langs = sorted(str(lang) for lang in (languages or LANGUAGES))
        seen_asr: set[str] = set()

        for lang in langs:
            segments = read_segments(source_root, lang, split)
            if limit is not None:
                segments = segments[:limit]
            wav_dir = (_pair_dir(source_root, lang, split) / "wav").resolve()

            groups: dict[str, list[Utterance]] = defaultdict(list)
            last_end: dict[Path, float] = {}
            for segment in segments:
                wav_path = wav_dir / segment["wav"]
                wav_id = f"{PREFIX}_{segment['wav'].removesuffix('.wav')}"
                start = segment["offset"]
                end = start + segment["duration"]
                last_end[wav_path] = max(last_end.get(wav_path, 0.0), end)

                shared = dict(
                    utt_id="",
                    wav_id=wav_id,
                    wav_path=str(wav_path),
                    start_time=start,
                    end_time=end,
                    lang=f"<{LANG}>",
                )
                groups[f"{segment['wav']}.asr"].append(
                    Utterance(
                        **shared,
                        task="<asr>",
                        text=segment["src"],
                        asr_text=segment["src"],
                    )
                )
                groups[f"{segment['wav']}.st_{lang}"].append(
                    Utterance(
                        **shared,
                        task=f"<st_{lang}>",
                        text=segment["tgt"],
                        asr_text=segment["src"],
                    )
                )

            errors = dict(
                run_parallel(
                    _verify_wav,
                    [(p, end, EXPECTED_FS) for p, end in last_end.items()],
                )
            )

            for group_name, group in groups.items():
                is_asr = group_name.endswith(".asr")
                error = errors.get(str(group[0].wav_path))
                if error is not None:
                    if failures is not None:
                        failures.write(
                            json.dumps({"utt_id": group[0].wav_id, "error": error})
                            + "\n"
                        )
                    continue

                for span in generate_long_utterances(group):
                    if is_asr:
                        if span.utt_id in seen_asr:
                            continue
                        seen_asr.add(span.utt_id)
                    yield {
                        "utt_id": span.utt_id,
                        "wav_path": span.wav_path,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                        "lang": LANG,
                        "task": "asr" if is_asr else "st",
                        "tgt_lang": "" if is_asr else lang,
                        "text": span.text_with_time,
                        "text_prev": span.prev_text,
                        "text_ctc": span.asr_text,
                    }
