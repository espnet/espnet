"""MuST-C speech translation dataset builder.

Reads the raw MuST-C v1 release directly (``<lang-pair>/data/<split>/{wav,txt}``)
instead of a Kaldi ``wav.scp``/``text`` directory prepared by
``egs2/must_c/st1/local/data.sh``. This recipe has no separate build step: the
builder's only job is to confirm the raw corpus is present and has the
expected layout for the configured language pair.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import yaml

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults


def _load_config() -> dict:
    """Load this package's config.yaml with its defaults resolved."""
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)


_CONFIG = _load_config()
_CFG = _CONFIG["builder"]
SRC_LANG = str(_CFG["src_lang"])
TGT_LANG = str(_CFG["tgt_lang"])

logger = logging.getLogger(__name__)
VERSION = str(_CFG["version"])
LANG_PAIR = f"{SRC_LANG}-{TGT_LANG}"
REQUIRED_SPLITS: tuple[str, ...] = tuple(str(s) for s in _CFG["required_splits"])
SOURCE_ENV_VAR = str(_CFG["source_env_var"])
DATASET_PATH = str(_CFG.get("dataset_path", "download"))

# "test" is the logical name; tst-COMMON is the directory on disk.
SPLIT_ALIASES: dict[str, str] = {
    str(k): str(v) for k, v in _CONFIG["dataset"]["split_aliases"].items()
}


def _resolve_tgt_lang(tgt_lang: str | None) -> str:
    """Return the target language, warning when the config default is taken.

    The default is ``all``, which indexes every installed ``en-<tgt>`` pair --
    14 of them for MuST-C v1.2, so dev becomes 17,740 examples instead of
    en-de's 1,423. That is a surprising thing to get by omission, hence the
    warning; pass ``tgt_lang="de"`` to reproduce egs2.
    """
    if tgt_lang:
        return str(tgt_lang)
    logger.warning(
        "tgt_lang was not given; using the dataset/config.yaml default %r. "
        "Pass tgt_lang='de' to reproduce egs2/must_c/st1.",
        TGT_LANG,
    )
    return TGT_LANG


def iter_source_candidates(
    recipe_root: Path,
    source_dir: str | Path | None,
) -> Iterable[Path]:
    """Yield candidate roots that may contain the raw MuST-C corpus."""
    if source_dir is not None:
        yield Path(source_dir)
    env_path = os.environ.get(SOURCE_ENV_VAR)
    if env_path:
        yield Path(env_path)
    yield recipe_root / DATASET_PATH
    yield recipe_root / "data"


def _lang_pair_dir(candidate: Path, tgt_lang: str = TGT_LANG) -> Path | None:
    """Return the ``en-<tgt_lang>`` directory under ``candidate``, if any."""
    lang_pair = f"{SRC_LANG}-{tgt_lang}"
    direct = candidate / lang_pair
    if direct.is_dir():
        return direct
    if candidate.name == lang_pair and candidate.is_dir():
        return candidate
    return None


def missing_required_splits(
    lang_pair_root: Path, tgt_lang: str = TGT_LANG
) -> list[str]:
    """Return required split names whose yaml/txt/wav files are incomplete."""
    missing = []
    for split in REQUIRED_SPLITS:
        split_dir = lang_pair_root / "data" / split
        # txt/<split>.yaml, txt/<split>.<src_lang>, txt/<split>.<tgt_lang>
        yaml_path = split_dir / "txt" / f"{split}.yaml"
        src_text = split_dir / "txt" / f"{split}.{SRC_LANG}"
        tgt_text = split_dir / "txt" / f"{split}.{tgt_lang}"
        wav_dir = split_dir / "wav"
        if not (
            yaml_path.is_file()
            and src_text.is_file()
            and tgt_text.is_file()
            and wav_dir.is_dir()
        ):
            missing.append(split)
    return missing


def resolve_source_root(
    recipe_root: Path,
    source_dir: str | Path | None = None,
    tgt_lang: str = TGT_LANG,
) -> Path:
    """Resolve the ``<...>/en-<tgt_lang>`` directory for this recipe's corpus."""
    checked: list[str] = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        checked.append(str(candidate))
        lang_pair_root = _lang_pair_dir(candidate, tgt_lang)
        if lang_pair_root is not None:
            return lang_pair_root

    raise FileNotFoundError(
        f"MuST-C {VERSION} language pair '{SRC_LANG}-{tgt_lang}' not found. "
        "Checked these locations:\n"
        + "\n".join(f"  - {path}/{SRC_LANG}-{tgt_lang}" for path in checked)
        + "\n"
        f"Unpack the corpus under '{DATASET_PATH}/' in the recipe, set "
        f"{SOURCE_ENV_VAR} to the raw corpus root (the directory that "
        f"contains '{LANG_PAIR}/'), or pass source_dir explicitly."
    )


def available_target_languages(
    recipe_root: Path, source_dir: str | Path | None = None
) -> list[str]:
    """Return every complete English-to-target MuST-C pair installed locally."""
    targets: set[str] = set()
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if candidate.is_dir():
            for pair in candidate.glob(f"{SRC_LANG}-*"):
                if pair.is_dir():
                    target = pair.name.removeprefix(f"{SRC_LANG}-")
                    if not missing_required_splits(pair, target):
                        targets.add(target)
    return sorted(targets)


class MustCSTBuilder(DatasetBuilder):
    """Validate raw MuST-C source availability for the recipe.

    This recipe reads the original MuST-C directory layout
    (``<lang-pair>/data/<split>/{wav,txt}``) directly during training and
    inference, so the builder's only responsibility is to ensure the expected
    split directories are available under the configured source root.
    """

    source_env_var = SOURCE_ENV_VAR

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        tgt_lang: str | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the required MuST-C splits are available."""
        recipe_root = Path(recipe_dir).resolve()
        try:
            selected = _resolve_tgt_lang(tgt_lang)
            targets = (
                available_target_languages(recipe_root, source_dir)
                if selected == "all"
                else [selected]
            )
            return bool(targets) and all(
                not missing_required_splits(
                    resolve_source_root(recipe_root, source_dir, item), item
                )
                for item in targets
            )
        except FileNotFoundError:
            return False

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        tgt_lang: str | None = None,
        **_kwargs,
    ) -> None:
        """Validate that the raw MuST-C source tree is already available.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional override pointing at the raw corpus root.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            FileNotFoundError: If the language-pair directory or required
                splits are missing.
        """
        recipe_root = Path(recipe_dir).resolve()
        selected = _resolve_tgt_lang(tgt_lang)
        targets = (
            available_target_languages(recipe_root, source_dir)
            if selected == "all"
            else [selected]
        )
        if not targets:
            raise FileNotFoundError(
                f"No complete MuST-C {SRC_LANG}-<target> pairs found"
            )
        # Validate the pinned target too: without this, prepare_source could
        # return while is_source_prepared stayed False.
        for target in targets:
            lang_pair_root = resolve_source_root(recipe_root, source_dir, target)
            missing = missing_required_splits(lang_pair_root, target)
            if missing:
                raise FileNotFoundError(
                    f"MuST-C {SRC_LANG}-{target} is missing required splits "
                    f"{missing} under {lang_pair_root}"
                )

    def is_built(self, recipe_dir, cache=None, **kwargs):
        """Whether the HF cache exists with the columns the Dataset reads.

        A cheap filesystem check, as ``DatasetBuilder`` asks for: source
        readiness is ``is_source_prepared``'s job, and the framework runs
        ``prepare_source`` before it ever calls ``build``.
        """
        cache_root = _hf_cache_root(recipe_dir, cache)
        if cache_root is None:
            return False
        return all(_cache_has_columns(cache_root / split) for split in _HF_CACHE_SPLITS)

    def build(self, recipe_dir, cache=None, **kwargs):
        """Write one HF cache split per required split."""
        cache_root = _hf_cache_root(recipe_dir, cache)
        if cache_root is None:
            raise RuntimeError(
                "must_c/esp2_st reads its splits from an HF cache, so the dataset "
                "cache must be enabled. Set `cache.enabled: true` (and a "
                "`cache.cache_dir`) in the training config."
            )
        _build_hf_cache(recipe_dir, cache_root, kwargs)


_HF_CACHE_SPLITS = ["train", "dev", "test", "tst-HE"]
_HF_CACHE_REQUIRED_COLUMNS = {
    "audio_path",
    "src_text",
    "tgt_text",
    "offset",
    "duration",
}


def _cache_has_columns(path):
    """Whether an HF cache split exists and carries the required columns."""
    import json

    try:
        features = json.loads((path / "dataset_info.json").read_text())["features"]
    except (OSError, KeyError, ValueError):
        return False
    return _HF_CACHE_REQUIRED_COLUMNS.issubset(features)


def _hf_cache_root(recipe_dir, cache):
    """Resolve the HF cache root, or None when caching is disabled."""
    if not cache or not cache.get("enabled", False):
        return None
    from pathlib import Path

    root = Path(cache.get("cache_dir", "data/hf"))
    if not root.is_absolute():
        root = Path(recipe_dir) / root
    return root / "hf_audio_index"


# Reading the raw corpus. These live here rather than in dataset.py so that
# nothing in this module imports dataset.py -- that cycle is what previously
# forced lazy imports.

_COMPACT_SEGMENT_RE = re.compile(
    r"^-\s*\{\s*duration:\s*([^,}]+),\s*offset:\s*([^,}]+),"
    r".*?speaker_id:\s*([^,}]+),\s*wav:\s*([^,}]+)\s*\}\s*$"
)


@dataclass(frozen=True)
class MustCExample:
    """Internal index entry for one MuST-C segment."""

    utt_id: str
    wav_path: Path
    offset: float
    duration: float
    speaker_id: str
    src_text: str
    tgt_text: str
    src_lang: str
    tgt_lang: str


def _parse_segments(split_dir: Path, split: str) -> list[tuple[float, float, str, str]]:
    """Parse ``txt/<split>.yaml`` into (offset, duration, speaker_id, wav) tuples."""
    yaml_path = split_dir / "txt" / f"{split}.yaml"
    segments: list[tuple[float, float, str, str]] = []
    with yaml_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            match = _COMPACT_SEGMENT_RE.match(line)
            if match is not None:
                duration, offset, speaker_id, wav = match.groups()
                segments.append(
                    (float(offset), float(duration), speaker_id.strip(), wav.strip())
                )
                continue
            try:
                entry = yaml.safe_load(line)
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"Unrecognized MuST-C yaml entry in {yaml_path}: {line}"
                ) from exc
            # A line beginning with ``-`` is parsed as a one-item sequence by
            # PyYAML; unwrap that sequence to its mapping entry.
            if isinstance(entry, list) and len(entry) == 1:
                entry = entry[0]
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Unrecognized MuST-C yaml entry in {yaml_path}: {line}"
                )
            required = {"duration", "offset", "speaker_id", "wav"}
            if not required.issubset(entry):
                raise ValueError(
                    f"Unrecognized MuST-C yaml entry in {yaml_path}: {line}"
                )
            try:
                duration = float(entry["duration"])
                offset = float(entry["offset"])
                speaker_id = str(entry["speaker_id"]).strip()
                wav = str(entry["wav"]).strip()
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid MuST-C yaml entry in {yaml_path}: {line}"
                ) from exc
            segments.append((offset, duration, speaker_id, wav))
    return segments


def _read_lines(path: Path) -> list[str]:
    """Read a text file into a list of lines, without trailing newlines."""
    with path.open("r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh]


def scan_split(
    lang_pair_root: Path, alias: str, tgt_lang: str = TGT_LANG
) -> list[MustCExample]:
    """Build an index for one split by zipping the yaml, en, and tgt files."""
    split_dir = lang_pair_root / "data" / alias
    wav_dir = split_dir / "wav"

    segments = _parse_segments(split_dir, alias)
    src_lines = _read_lines(split_dir / "txt" / f"{alias}.{SRC_LANG}")
    tgt_lines = _read_lines(split_dir / "txt" / f"{alias}.{tgt_lang}")

    if not (len(segments) == len(src_lines) == len(tgt_lines)):
        raise RuntimeError(
            f"MuST-C {alias}: yaml/{SRC_LANG}/{tgt_lang} line counts differ "
            f"({len(segments)}, {len(src_lines)}, {len(tgt_lines)})"
        )

    examples: list[MustCExample] = []
    talk_counters: dict[str, int] = {}
    for (offset, duration, speaker_id, wav_name), src_text, tgt_text in zip(
        segments, src_lines, tgt_lines
    ):
        talk_id = Path(wav_name).stem
        idx = talk_counters.get(talk_id, 0)
        talk_counters[talk_id] = idx + 1
        utt_id = f"{alias}_{talk_id}_{idx:04d}"
        examples.append(
            MustCExample(
                utt_id=utt_id,
                wav_path=(wav_dir / wav_name).resolve(),
                offset=offset,
                duration=duration,
                speaker_id=speaker_id,
                src_text=src_text,
                tgt_text=tgt_text,
                src_lang=SRC_LANG,
                tgt_lang=tgt_lang,
            )
        )

    if not examples:
        raise RuntimeError(f"No segments found for MuST-C split: {split_dir}")
    return examples


@lru_cache(maxsize=None)
def _wav_samplerate(wav_path: str) -> int:
    """Sample rate of one wav, cached because talks are read many times."""
    return int(sf.info(wav_path).samplerate)


def read_segment(wav_path: Path, offset: float, duration: float) -> np.ndarray:
    """Read one segment out of a talk-length wav as float32."""
    samplerate = _wav_samplerate(str(wav_path))
    start = int(round(offset * samplerate))
    frames = int(round(duration * samplerate))
    array, _sr = sf.read(
        str(wav_path), start=start, frames=frames, dtype="float32", always_2d=False
    )
    return np.asarray(array, dtype=np.float32)


def _verify_segment(task):
    """Decode one segment to confirm it is readable.

    Module-level and argument-only so it can be pickled to a Dask worker; it
    must not close over the Dataset, which is not serializable.

    Args:
        task: ``(index, wav_path, offset, duration)``.

    Returns:
        ``(index, None)`` when the segment decodes to finite, non-empty audio,
        otherwise ``(index, "<ExcType>: <message>")``.
    """
    index, wav_path, offset, duration = task
    try:
        speech = np.asarray(read_segment(wav_path, offset, duration))
        if speech.size == 0 or not np.isfinite(speech).all():
            raise ValueError("decoded audio is empty or non-finite")
    except Exception as exc:  # noqa: BLE001 - recorded per segment, not raised
        return index, repr(exc)
    return index, None


def _verify_segments(tasks):
    """Verify every segment, in parallel when a parallel config is set.

    Decoding each of MuST-C's ~230k train segments is the whole cost of a cache
    build and the segments are independent, so this fans out over
    ``espnet3.parallel``. Recipes that never called ``set_parallel`` -- or a
    plain ``Dataset(...)`` call outside a stage -- fall back to the serial path
    rather than failing.

    Args:
        tasks: Sequence of ``(index, wav_path, offset, duration)``.

    Returns:
        ``{index: error_or_None}`` for every task.
    """
    tasks = list(tasks)
    if not tasks:
        return {}
    # Deferred: espnet3.parallel.parallel builds CLUSTER_MAP at module scope,
    # so importing it pulls dask and distributed -- 4s that a read-only
    # `import ...dataset` should not pay. The submodule, not the package:
    # espnet3/parallel/__init__.py re-exports nothing.
    from espnet3.parallel.parallel import get_client, get_parallel_config

    config = get_parallel_config()
    env = getattr(config, "env", "local") if config is not None else "local"

    if env != "local":
        logger.info("verifying %d segments on the %s cluster", len(tasks), env)
        with get_client(config) as client:
            return dict(client.gather(client.map(_verify_segment, tasks)))

    # espnet3 reads `env: local` as "no Dask cluster" and runs in-process
    # (base_runner.py: _run_local), so building a LocalCluster here would both
    # break that convention and fail wherever workers cannot be spawned.
    # Decoding is IO-bound and soundfile releases the GIL, so a thread pool
    # gives the speedup without a cluster.
    workers = int(getattr(config, "n_workers", 1) or 1) if config is not None else 1
    if workers <= 1:
        logger.info("verifying %d segments serially", len(tasks))
        return dict(_verify_segment(task) for task in tasks)

    from concurrent.futures import ThreadPoolExecutor

    logger.info("verifying %d segments on %d local threads", len(tasks), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_verify_segment, tasks))


def _build_hf_cache(recipe_dir, cache_root, dataset_kwargs):
    """Write one HF cache split per required split, skipping complete ones."""
    import json
    import shutil

    from datasets import Dataset as HFDataset

    recipe_root = Path(recipe_dir).resolve()
    tgt_lang = _resolve_tgt_lang(dataset_kwargs.get("tgt_lang"))
    lang_pair_root = resolve_source_root(
        recipe_root, source_dir=dataset_kwargs.get("source_dir"), tgt_lang=tgt_lang
    )

    cache_root.mkdir(parents=True, exist_ok=True)
    for split in _HF_CACHE_SPLITS:
        target = cache_root / split
        if _cache_has_columns(target):
            continue
        if target.is_dir():
            shutil.rmtree(target)
        temporary = cache_root / f".{split}.tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        failures = cache_root / f"{split}.failures.jsonl"

        def rows():
            """Yield one cache row per corpus segment, logging failures."""
            # The cache is the corpus as released; the filter is a read-path
            # concern and must not reach it.
            examples = scan_split(
                lang_pair_root, SPLIT_ALIASES.get(split, split), tgt_lang
            )
            # Decoded audio is never stored, so this pass only proves each
            # segment is readable; the rows below are pure metadata.
            errors = _verify_segments(
                (index, example.wav_path, example.offset, example.duration)
                for index, example in enumerate(examples)
            )
            with failures.open("w", encoding="utf-8") as stream:
                for index, example in enumerate(examples):
                    try:
                        error = errors.get(index)
                        if error is not None:
                            raise RuntimeError(error)
                        yield {
                            "raw_index": index,
                            "audio_path": str(example.wav_path),
                            "utt_id": str(example.utt_id),
                            # Unused by the reader; kept so a rebuilt cache
                            # matches the columns of existing ones.
                            "text": str(example.tgt_text),
                            "src_text": str(example.src_text),
                            "tgt_text": str(example.tgt_text),
                            "src_lang": str(example.src_lang),
                            "tgt_lang": str(example.tgt_lang),
                            "lang_pair": f"{example.src_lang}-{example.tgt_lang}",
                            "offset": float(example.offset),
                            "duration": float(example.duration),
                        }
                    except Exception as exc:
                        stream.write(
                            json.dumps({"raw_index": index, "error": repr(exc)}) + "\n"
                        )

        HFDataset.from_generator(rows).save_to_disk(str(temporary))
        temporary.rename(target)
