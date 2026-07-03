"""LibriTTS dataset builder for the ESPnet3 codec recipe.

Audio-only manifest builder: unlike the TTS recipe, codec training needs no
transcript or speaker-id column, so the manifest here is a plain
``utt_id<TAB>wav_path`` TSV.
"""

from __future__ import annotations

import logging
import subprocess
from importlib import resources
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def _scan_subset_entries(subset_dir: Path) -> list[tuple[str, Path]]:
    """Scan a subset directory and return ``(utt_id, wav_path)`` tuples.

    Args:
        subset_dir: Path to the subset directory (e.g.,
            ``LibriTTS/train-clean-100``).

    Returns:
        List of ``(utt_id, wav_path)`` tuples for every utterance with both a
        transcript and a wav file present. The transcript is only used to
        confirm the utterance is well-formed; its text is not carried into
        the manifest since codec training does not consume text.
    """
    entries = []
    for text_path in sorted(subset_dir.rglob("*.normalized.txt")):
        wav_path = text_path.with_suffix("").with_suffix(".wav")
        if not wav_path.is_file():
            continue
        if not text_path.read_text(encoding="utf-8").strip():
            continue
        # LibriTTS's native utt_id is underscore-separated digit groups (e.g.
        # "1089_134691_000004_000001"), which Python's int() silently accepts
        # via PEP 515 digit-group underscores. That trips up
        # espnet3.components.data.dataset.CombinedDataset.__getitem__, which
        # tries int(idx) on string keys from ESPnet's chunk/sequence
        # iterators to decide int-vs-utterance-id indexing. Using hyphens
        # (LibriSpeech's own convention) keeps the ID human-readable while
        # guaranteeing int() raises, so lookups always take the intended
        # utterance-ID path instead of being misparsed as a huge integer.
        utt_id = text_path.stem.replace(".normalized", "").replace("_", "-")
        entries.append((utt_id, wav_path.resolve()))
    return entries


class LibriTTSCodecBuilder(DatasetBuilder):
    """Prepare LibriTTS audio-only manifests for ESPnet3 codec training."""

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        **_kwargs,
    ) -> bool:
        """Check if LibriTTS source data is prepared."""
        recipe_root = Path(recipe_dir).resolve()
        libritts_root = recipe_root / _CFG["dataset_path"] / "LibriTTS"
        required = []
        for subsets in _CFG["split_subsets"].values():
            required.extend(subsets)
        return all((libritts_root / subset).is_dir() for subset in required)

    def prepare_source(
        self,
        recipe_dir: str | Path,
        **_kwargs,
    ) -> None:
        """Prepare LibriTTS source data by downloading if necessary.

        Raises:
            RuntimeError: If a subset download fails.
        """
        dataset_root = Path(recipe_dir).resolve() / _CFG["dataset_path"]

        if self.is_source_prepared(recipe_dir=recipe_dir):
            logger.info("LibriTTS source data is already prepared, skipping download.")
            return

        dataset_root.mkdir(parents=True, exist_ok=True)
        script_path = Path(recipe_dir).resolve() / "local" / "download_libritts.sh"

        required_subsets = []
        for subsets in _CFG["split_subsets"].values():
            required_subsets.extend(subsets)
        for subset in required_subsets:
            subset_marker = dataset_root / "LibriTTS" / subset / ".complete"
            if subset_marker.is_file():
                logger.info(f"Subset {subset} already downloaded, skipping.")
                continue
            logger.info(f"Downloading LibriTTS subset: {subset}")
            try:
                subprocess.run(
                    ["bash", str(script_path), str(dataset_root), subset],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to download LibriTTS subset {subset}. "
                    "Check internet connection and disk space."
                ) from e

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check if LibriTTS codec manifests are built."""
        recipe_root = Path(recipe_dir).resolve()
        data_dir = recipe_root / _CFG["data_path"]
        return all(
            (data_dir / relpath).is_file()
            for relpath in _CFG["manifest_paths"].values()
        )

    def build(
        self,
        recipe_dir: str | Path,
        **_kwargs,
    ) -> None:
        """Build LibriTTS codec manifests (audio-only).

        Raises:
            FileNotFoundError: If a configured subset directory is missing.
        """
        recipe_root = Path(recipe_dir).resolve()
        libritts_root = recipe_root / _CFG["dataset_path"] / "LibriTTS"
        data_dir = recipe_root / _CFG["data_path"]
        data_dir.mkdir(parents=True, exist_ok=True)

        for split, subsets in _CFG["split_subsets"].items():
            entries = []
            for subset in subsets:
                subset_dir = libritts_root / subset
                if not subset_dir.is_dir():
                    raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
                entries.extend(_scan_subset_entries(subset_dir))
            entries = sorted(entries, key=lambda x: x[0])

            manifest_path = data_dir / _CFG["manifest_paths"][split]
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w", encoding="utf-8") as f:
                for utt_id, wav_path in entries:
                    f.write(f"{utt_id}\t{wav_path}\n")
