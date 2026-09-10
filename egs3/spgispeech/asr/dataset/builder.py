"""SPGISpeech dataset builder.

This recipe reads the raw SPGISpeech distribution directly (the
``train.csv``/``val.csv`` manifests plus the ``spgispeech/{train,val}/``
audio tree) instead of a Kaldi data directory produced by
``egs2/spgispeech/asr1/local/data.sh``. The builder's only job is therefore
to locate and validate that raw corpus tree; there is no build/caching step.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Iterable

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults


def _load_builder_config() -> dict:
    """Return the ``builder``/``dataset`` sections of ``dataset/config.yaml``."""
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def _csv_files(source_root: Path) -> dict[str, Path]:
    """Map each base split to its manifest csv under ``source_root``."""
    return {
        base_split: source_root / str(csv_name)
        for base_split, csv_name in _CFG["csv_files"].items()
    }


def _audio_dirs(source_root: Path) -> dict[str, Path]:
    """Map each base split to its audio directory under ``source_root``."""
    audio_subdir = str(_CFG["audio_subdir"])
    return {
        base_split: source_root / audio_subdir / base_split
        for base_split in _CFG["csv_files"]
    }


def _is_valid_source_root(candidate: Path) -> bool:
    """Return whether ``candidate`` holds every required csv and audio directory."""
    if not candidate.is_dir():
        return False
    csvs = _csv_files(candidate)
    audio_dirs = _audio_dirs(candidate)
    return all(path.is_file() for path in csvs.values()) and all(
        path.is_dir() for path in audio_dirs.values()
    )


def iter_source_candidates(
    recipe_root: Path,
    source_dir: str | Path | None,
) -> Iterable[Path]:
    """Yield candidate directories that may be the SPGISpeech root.

    Ordered by precedence: an explicit ``source_dir`` first, so a caller that
    names a corpus root is never silently overridden by a recipe-local copy,
    then ``<recipe_dir>/download/spgispeech``, then the environment variable.
    """
    if source_dir is not None:
        yield Path(source_dir)

    yield recipe_root / _CFG["dataset_path"] / "spgispeech"

    env_var = str(_CFG["source_env_var"])
    env_path = os.environ.get(env_var)
    if env_path:
        yield Path(env_path)

    default_dir = _CFG.get("default_source_dir")
    if default_dir:
        yield Path(str(default_dir))


def resolve_source_root(
    recipe_root: Path,
    source_dir: str | Path | None = None,
) -> Path:
    """Resolve the usable SPGISpeech source root for this recipe.

    The expected on-disk layout (matching the raw corpus distribution) is::

        <source_root>/
            train.csv
            val.csv
            spgispeech/train/<hash>/<n>.wav
            spgispeech/val/<hash>/<n>.wav
    """
    checked: list[str] = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        checked.append(str(candidate))
        if _is_valid_source_root(candidate):
            return candidate.resolve()

    env_var = str(_CFG["source_env_var"])
    raise FileNotFoundError(
        "SPGISpeech source not found. Checked these locations:\n"
        + "\n".join(f"  - {path}" for path in checked)
        + "\n"
        + f"Set {env_var} to the corpus root, or place it under "
        + "<recipe_dir>/download/spgispeech."
    )


class SPGISpeechBuilder(DatasetBuilder):
    """Validate SPGISpeech source availability for the recipe.

    This recipe reads the raw ``train.csv``/``val.csv`` manifests and audio
    tree directly during training and inference, so the builder's only
    responsibility is to ensure the expected files are available under the
    configured source directory (or the ``SPGISPEECH`` environment variable,
    or the shared corpus default).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the SPGISpeech raw corpus tree is available."""
        recipe_root = Path(recipe_dir).resolve()
        try:
            resolve_source_root(recipe_root, source_dir=source_dir)
        except FileNotFoundError:
            return False
        return True

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Validate that the SPGISpeech source tree is already available.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional override pointing to the SPGISpeech root.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            FileNotFoundError: If the SPGISpeech root or required files are
                missing.
        """
        recipe_root = Path(recipe_dir).resolve()
        resolve_source_root(recipe_root, source_dir=source_dir)

    def _is_recipe_built(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return source readiness because this recipe has no build artifacts."""
        return self.is_source_prepared(
            recipe_dir=recipe_dir,
            source_dir=source_dir,
        )

    def _build_recipe_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """No-op build step for raw-corpus-backed SPGISpeech access."""
        self.prepare_source(recipe_dir=recipe_dir, source_dir=source_dir)

    def is_built(self, recipe_dir, cache=None, **kwargs):
        """Return whether every split in ``_HF_CACHE_SPLITS`` has been cached."""
        cache_root = _hf_cache_root(recipe_dir, cache)
        if cache_root is not None:
            return all((cache_root / split).is_dir() for split in _HF_CACHE_SPLITS)
        return _call_supported(
            self._is_recipe_built,
            recipe_dir=recipe_dir,
            cache=cache,
            **kwargs,
        )

    def build(self, recipe_dir, cache=None, **kwargs):
        """Build the audio index, preparing the raw source first if necessary."""
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


# Every split in dataset/config.yaml:supported_splits, because
# SPGISpeechDataset loads hf_audio_index/<split> whenever caching is enabled:
# a name missing here raises FileNotFoundError even after the builder has run.
# _build_hf_cache stores row.raw_text for the *_unnorm splits instead of
# applying normalize_text.
_HF_CACHE_SPLITS = [
    "val",
    "dev_4k",
    "train_nodev",
    "train",
    "val_unnorm",
    "dev_4k_unnorm",
    "train_nodev_unnorm",
    "train_unnorm",
]


def _hf_cache_root(recipe_dir, cache):
    """Return the ``hf_audio_index`` root for this cache config, or None if disabled."""
    if not cache or not cache.get("enabled", False):
        return None
    from pathlib import Path

    root = Path(cache.get("cache_dir", "data/hf"))
    if not root.is_absolute():
        root = Path(recipe_dir) / root
    return root / "hf_audio_index"


def _call_supported(function, **kwargs):
    """Call ``function`` with only the keyword arguments its signature accepts."""
    import inspect

    parameters = inspect.signature(function).parameters
    variadic = any(
        parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    clean = {
        key: value for key, value in kwargs.items() if variadic or key in parameters
    }
    return function(**clean)


def _raw_record(dataset, index):
    """Return the raw manifest row backing ``index`` in ``dataset``."""
    for name in ("_examples", "_rows", "_entries", "examples", "entries"):
        rows = getattr(dataset, name, None)
        if rows is not None:
            return rows[index]
    if hasattr(dataset, "_utt_ids") and hasattr(dataset, "_wav"):
        return dataset._wav[dataset._utt_ids[index]]
    return None


def _audio_path(record):
    """Extract the audio path from a manifest record, trying the known key names."""
    from dataclasses import asdict, is_dataclass
    from pathlib import Path

    if is_dataclass(record):
        record = asdict(record)
    if not isinstance(record, dict):
        record = getattr(record, "__dict__", {"audio_path": record})
    suffixes = {".wav", ".flac", ".mp3", ".opus", ".sph", ".pcm", ".m4a"}
    preferred = (
        "segment_path",
        "audio_path",
        "wav_path",
        "sph_path",
        "mp3_path",
        "long_path",
        "recording_path",
        "path",
        "speech",
    )
    values = [record.get(key) for key in preferred if record.get(key) is not None]
    values.extend(record.values())
    for value in values:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, str):
            stripped = value.rstrip("|").strip()
            if Path(stripped).suffix.lower() in suffixes or value.rstrip().endswith(
                "|"
            ):
                return value
    raise RuntimeError("Could not locate an audio path in the raw index record")


def _build_hf_cache(recipe_dir, cache_root, dataset_kwargs):
    """Write one HuggingFace audio index per split under ``cache_root``.

        Splits whose directory already exists are skipped, so the build is
        incremental. Each split is written to a temporary directory and renamed
        into place, so an interrupted run leaves no partial index behind.
        """
    import concurrent.futures
    import importlib
    import json
    import shutil

    from datasets import Dataset as HFDataset

    module = importlib.import_module(__package__)
    dataset_class = module.Dataset
    cache_root.mkdir(parents=True, exist_ok=True)
    for split in _HF_CACHE_SPLITS:
        target = cache_root / split
        if target.is_dir():
            continue
        temporary = cache_root / f".{split}.tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        failures = cache_root / f"{split}.failures.jsonl"

        # max_utts must not reach the cache builder: it truncates the split,
        # while is_built() only checks that the directory exists, so a later
        # full run would silently accept a partial cache.
        canonical_kwargs = dict(dataset_kwargs)
        canonical_kwargs.pop("max_utts", None)

        dataset = _call_supported(
            dataset_class,
            split=split,
            recipe_dir=recipe_dir,
            cache={"enabled": False},
            **canonical_kwargs,
        )
        dataset_impl = importlib.import_module(module.Dataset.__module__)
        normalize = dataset_impl.normalize_text
        records = [
            (
                i,
                str(row.audio_path),
                row.utt_id,
                (
                    normalize(row.raw_text)
                    if not split.endswith("_unnorm")
                    else row.raw_text
                ),
            )
            for i, row in enumerate(dataset._examples)
        ]

        def rows():
            """Yield one validated cache row per readable utterance."""
            with failures.open("w", encoding="utf-8") as stream:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(
                        16, int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
                    )
                ) as pool:
                    for result in pool.map(_validate_spgi_row, records, chunksize=64):
                        if result["ok"]:
                            i, path, utt_id, text = result["row"]
                            yield {
                                "raw_index": i,
                                "audio_path": path,
                                "utt_id": utt_id,
                                "text": text,
                                "src_text": "",
                                "tgt_text": "",
                                "lang": "en",
                            }
                        else:
                            stream.write(json.dumps(result) + "\n")

        HFDataset.from_generator(rows).save_to_disk(str(temporary))
        temporary.rename(target)


def _validate_spgi_row(record):
    """Check that one record's audio is readable, for use in a worker pool."""
    import numpy as np
    import soundfile as sf

    i, path, utt_id, text = record
    try:
        audio, _ = sf.read(path, dtype="float32")
        if not np.asarray(audio).size or not np.isfinite(audio).all() or not text:
            raise ValueError("invalid audio/text")
        return {"ok": True, "row": record}
    except Exception as exc:
        return {"ok": False, "raw_index": i, "error": repr(exc)}
