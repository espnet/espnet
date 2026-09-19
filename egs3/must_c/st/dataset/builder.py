"""MuST-C speech translation dataset builder.

Reads the raw MuST-C v1 release directly (``<lang-pair>/data/<split>/{wav,txt}``)
instead of a Kaldi ``wav.scp``/``text`` directory prepared by
``egs2/must_c/st1/local/data.sh``. This recipe has no separate build step: the
builder's only job is to confirm the raw corpus is present and has the
expected layout for the configured language pair.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Iterable

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
VERSION = str(_CFG["version"])
LANG_PAIR = f"{SRC_LANG}-{TGT_LANG}"
REQUIRED_SPLITS: tuple[str, ...] = tuple(str(s) for s in _CFG["required_splits"])
SOURCE_ENV_VAR = str(_CFG["source_env_var"])
SOURCE_DIR_DEFAULT = str(_CFG["source_dir_default"])

# Long/short filtering (egs2 st.sh stage 4). st.sh keeps the unfiltered index
# at ${data_feats}/org/<dset>; the HF cache here is that `org` side, and the
# Dataset applies the bounds when it reads it. See dataset/config.yaml.
_FILTER_CFG = _CONFIG["filter"]
MIN_WAV_DURATION = float(_FILTER_CFG["min_wav_duration"])
MAX_WAV_DURATION = float(_FILTER_CFG["max_wav_duration"])
FILTERED_SPLITS: tuple[str, ...] = tuple(str(s) for s in _FILTER_CFG["splits"])
FILTER_TOKENIZER_TEXT = bool(_FILTER_CFG["apply_to_tokenizer_text"])


def split_is_filtered(split: str) -> bool:
    """Whether ``split`` is one st.sh would trim (train and valid, not test)."""
    return str(split) in FILTERED_SPLITS


def keep_duration(duration: float) -> bool:
    """Reproduce st.sh's ``$2 > min_length && $2 < max_length``.

    Both bounds are strict, as in the awk expression st.sh applies to
    ``utt2num_samples``.
    """
    return MIN_WAV_DURATION < float(duration) < MAX_WAV_DURATION


def kept_indices(durations, split: str) -> list[int] | None:
    """Indices of ``durations`` to keep for ``split``.

    Args:
        durations: Segment durations in seconds, in corpus order.
        split: Logical split name, e.g. ``"train"`` or ``"test"``.

    Returns:
        The positions to keep, or ``None`` when ``split`` is not filtered at
        all. ``None`` rather than ``list(range(len(durations)))`` so callers
        can skip the indirection entirely on the test splits.
    """
    if not split_is_filtered(split):
        return None
    return [i for i, duration in enumerate(durations) if keep_duration(duration)]


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
    yield Path(SOURCE_DIR_DEFAULT)
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
        f"Set {SOURCE_ENV_VAR} to the raw corpus root (the directory that "
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
            selected = tgt_lang or TGT_LANG
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
        selected = tgt_lang or TGT_LANG
        targets = (
            available_target_languages(recipe_root, source_dir)
            if selected == "all"
            else [selected]
        )
        if not targets:
            raise FileNotFoundError(
                f"No complete MuST-C {SRC_LANG}-<target> pairs found"
            )
        # A pinned target was never checked, so prepare_source could return
        # while is_source_prepared stayed False. resolve_source_root raises if
        # the pair directory is absent; missing_required_splits catches a pair
        # that exists but is incomplete.
        for target in targets:
            lang_pair_root = resolve_source_root(recipe_root, source_dir, target)
            missing = missing_required_splits(lang_pair_root, target)
            if missing:
                raise FileNotFoundError(
                    f"MuST-C {SRC_LANG}-{target} is missing required splits "
                    f"{missing} under {lang_pair_root}"
                )

    def _is_recipe_built(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        tgt_lang: str | None = None,
        **_kwargs,
    ) -> bool:
        """Return source readiness because this recipe has no build artifacts."""
        return self.is_source_prepared(
            recipe_dir=recipe_dir, source_dir=source_dir, tgt_lang=tgt_lang
        )

    def _build_recipe_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        tgt_lang: str | None = None,
        **_kwargs,
    ) -> None:
        """No-op build step for raw-directory-backed MuST-C access."""
        self.prepare_source(
            recipe_dir=recipe_dir, source_dir=source_dir, tgt_lang=tgt_lang
        )

    def is_built(self, recipe_dir, cache=None, **kwargs):
        """Whether the cache (or, uncached, the raw source) is ready."""
        cache_root = _hf_cache_root(recipe_dir, cache)
        if cache_root is not None:
            return all(
                _cache_has_columns(cache_root / split) for split in _HF_CACHE_SPLITS
            )
        return _call_supported(
            self._is_recipe_built,
            recipe_dir=recipe_dir,
            cache=cache,
            **kwargs,
        )

    def build(self, recipe_dir, cache=None, **kwargs):
        """Build the HF cache, preparing the raw source first if needed."""
        cache_root = _hf_cache_root(recipe_dir, cache)
        if cache_root is None:
            return _call_supported(
                self._build_recipe_source,
                recipe_dir=recipe_dir,
                cache=cache,
                **kwargs,
            )
        no_cache = dict(cache)
        no_cache["enabled"] = False
        no_cache["backend"] = "omniio"
        if not _call_supported(
            self._is_recipe_built,
            recipe_dir=recipe_dir,
            cache=no_cache,
            **kwargs,
        ):
            _call_supported(
                self._build_recipe_source,
                recipe_dir=recipe_dir,
                cache=no_cache,
                **kwargs,
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


def _call_supported(function, **kwargs):
    """Call ``function`` with only the keyword arguments it accepts."""
    import inspect

    parameters = inspect.signature(function).parameters
    variadic = any(
        parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    clean = {
        key: value for key, value in kwargs.items() if variadic or key in parameters
    }
    return function(**clean)


def _build_hf_cache(recipe_dir, cache_root, dataset_kwargs):
    """Write one HF cache split per required split, skipping complete ones."""
    import importlib
    import json
    import shutil

    import numpy as np
    from datasets import Dataset as HFDataset

    module = importlib.import_module(__package__)
    dataset_class = module.Dataset
    # Deferred: dataset.py imports this module, so a top-level import loops.
    _read_segment = importlib.import_module(f"{__package__}.dataset")._read_segment
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
            dataset = _call_supported(
                dataset_class,
                split=split,
                recipe_dir=recipe_dir,
                cache={"enabled": False},
                # The cache is the corpus as released; the filter belongs to
                # the read path. Iterating `_examples` ignores it regardless.
                apply_filter=False,
                **dataset_kwargs,
            )
            if getattr(dataset, "_examples", None) is None:
                raise RuntimeError(
                    "cache build needs the raw-scan Dataset; got a cache-backed "
                    "one, which has no _examples (see Dataset.__init__)"
                )
            with failures.open("w", encoding="utf-8") as stream:
                # Every field, audio included, comes from `example`. Reading
                # audio via dataset[index] would mix index spaces: __getitem__
                # maps through `_keep`, `_examples[index]` does not.
                for index, example in enumerate(dataset._examples):
                    try:
                        speech = _read_segment(
                            example.wav_path, example.offset, example.duration
                        )
                        speech = np.asarray(speech)
                        if speech.size == 0 or not np.isfinite(speech).all():
                            raise ValueError("decoded audio is empty or non-finite")
                        # Text from `example` (raw corpus), never from
                        # __getitem__ output: that is already case-folded and
                        # carries only speech/text/src_text.
                        yield {
                            "raw_index": index,
                            "audio_path": str(example.wav_path),
                            "utt_id": str(example.utt_id),
                            # Unused by the reader; kept so a rebuilt cache
                            # keeps the same columns as existing ones. For
                            # task="st" the target side is the corpus text.
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
