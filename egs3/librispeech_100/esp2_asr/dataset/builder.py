"""Build Hugging Face Dataset indexes over the shared LibriSpeech corpus."""

from __future__ import annotations

import os
import shutil
from importlib import resources
from pathlib import Path
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults


def _load_dataset_config() -> dict:
    """Load recipe-local dataset settings."""
    resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)


_CONFIG = _load_dataset_config()
_BUILDER_CONFIG = _CONFIG["builder"]
_DATASET_CONFIG = _CONFIG["dataset"]
_REQUIRED_SPLITS = tuple(str(split) for split in _BUILDER_CONFIG["required_splits"])


def _resolve_librispeech_root(data_dir: str | Path) -> Path:
    """Resolve a parent directory or direct LibriSpeech root."""
    candidate = Path(data_dir).expanduser()
    if (candidate / "LibriSpeech").is_dir():
        return candidate / "LibriSpeech"
    if candidate.name == "LibriSpeech" and candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"LibriSpeech root not found under {candidate}")


def resolve_source_root(
    recipe_dir: str | Path, source_dir: str | Path | None = None
) -> Path:
    """Resolve the corpus root from config override, environment, or recipe."""
    recipe_root = Path(recipe_dir).resolve()
    candidates = [recipe_root / str(_BUILDER_CONFIG["dataset_path"])]
    if source_dir is not None:
        candidates.append(Path(source_dir).expanduser())
    env_path = os.environ.get(str(_BUILDER_CONFIG["source_env_var"]))
    if env_path:
        candidates.append(Path(env_path).expanduser())

    checked = []
    for candidate in candidates:
        checked.append(str(candidate))
        try:
            return _resolve_librispeech_root(candidate)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "LibriSpeech source not found. Checked:\n"
        + "\n".join(f"  - {path}" for path in checked)
        + f"\nSet {_BUILDER_CONFIG['source_env_var']} to the corpus root."
    )


def _iter_split_rows(split_dir: Path) -> Iterator[dict[str, str]]:
    """Yield audio path and transcript records from LibriSpeech transcript files."""
    # LibriSpeech is laid out as split/speaker/chapter/*.trans.txt. A recursive
    # walk also stats every audio file on shared filesystems, so stay at the
    # known transcript depth.
    for transcript_path in sorted(split_dir.glob("*/*/*.trans.txt")):
        for raw_line in transcript_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            utterance_id, separator, text = line.partition(" ")
            if not separator or not text.strip():
                continue
            audio_path = transcript_path.parent / f"{utterance_id}.flac"
            # ``split_dir`` is resolved from the recipe/source root, so this
            # path is absolute already. Avoid ``Path.resolve()`` here: on NFS
            # it probes every waveform while building the lightweight index.
            yield {"audio": str(audio_path), "text": text.strip()}


def _resolve_cache_dir(recipe_dir: str | Path) -> Path:
    """Resolve the recipe-local Hugging Face dataset cache directory."""
    return Path(recipe_dir).resolve() / str(_BUILDER_CONFIG["dataset_cache_dir"])


class LibriSpeech100Builder(DatasetBuilder):
    """Index LibriSpeech audio paths and transcripts as saved HF Datasets."""

    def is_source_prepared(
        self, recipe_dir: str | Path, source_dir: str | Path | None = None, **_kwargs
    ) -> bool:
        """Check whether every required raw corpus split is available.

        Args:
            recipe_dir: LibriSpeech recipe root.
            source_dir: Optional parent or direct LibriSpeech directory.
            **_kwargs: Ignored builder options accepted by the shared interface.

        Returns:
            bool: True when all required LibriSpeech split directories exist.

        Example:
            >>> LibriSpeech100Builder().is_source_prepared("recipe")
            False
        """
        try:
            root = resolve_source_root(recipe_dir, source_dir)
        except FileNotFoundError:
            return False
        return all((root / split).is_dir() for split in _REQUIRED_SPLITS)

    def prepare_source(
        self, recipe_dir: str | Path, source_dir: str | Path | None = None, **_kwargs
    ) -> None:
        """Validate the shared raw corpus location without copying the audio.

        Args:
            recipe_dir: LibriSpeech recipe root.
            source_dir: Optional parent or direct LibriSpeech directory.
            **_kwargs: Ignored builder options accepted by the shared interface.

        Returns:
            None: Raises an actionable error when the corpus is incomplete.

        Example:
            >>> LibriSpeech100Builder().prepare_source("egs3/librispeech_100/esp2_asr")
        """
        root = resolve_source_root(recipe_dir, source_dir)
        missing = [split for split in _REQUIRED_SPLITS if not (root / split).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"LibriSpeech source at {root} is missing: {', '.join(missing)}"
            )

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether all recipe-local HF split indexes have been written.

        Args:
            recipe_dir: LibriSpeech recipe root.
            **_kwargs: Ignored builder options accepted by the shared interface.

        Returns:
            bool: True when each required split has saved HF dataset metadata.

        Example:
            >>> LibriSpeech100Builder().is_built("egs3/librispeech_100/esp2_asr")
            False
        """
        cache_dir = _resolve_cache_dir(recipe_dir)
        return all(
            (cache_dir / split / "dataset_info.json").is_file()
            and (cache_dir / split / "state.json").is_file()
            for split in _REQUIRED_SPLITS
        )

    def build(
        self, recipe_dir: str | Path, source_dir: str | Path | None = None, **_kwargs
    ) -> None:
        """Save one HF Dataset per split with only audio paths and transcripts.

        Args:
            recipe_dir: LibriSpeech recipe root; cache is created under ``data/``.
            source_dir: Optional parent or direct LibriSpeech directory.
            **_kwargs: Ignored builder options accepted by the shared interface.

        Returns:
            None: Saved datasets contain only string columns ``audio`` and ``text``.

        Example:
            >>> LibriSpeech100Builder().build("egs3/librispeech_100/esp2_asr")
        """
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError(
                "LibriSpeech cache creation requires Hugging Face Datasets. "
                "Install the ESPnet training extra with `pip install 'espnet[train]'`."
            ) from exc

        root = resolve_source_root(recipe_dir, source_dir)
        self.prepare_source(recipe_dir, source_dir)
        cache_dir = _resolve_cache_dir(recipe_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for split in _REQUIRED_SPLITS:
            target = cache_dir / split
            if (target / "dataset_info.json").is_file() and (
                target / "state.json"
            ).is_file():
                continue
            if target.exists():
                shutil.rmtree(target)
            rows = list(_iter_split_rows(root / split))
            if not rows:
                raise RuntimeError(
                    f"No LibriSpeech audio/transcript pairs in {root / split}"
                )
            Dataset.from_list(rows).save_to_disk(str(target))
