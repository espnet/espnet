"""LibriTTS dataset builder for ESPnet3 TTS recipe."""

from __future__ import annotations

import logging
import subprocess
from importlib import resources
from pathlib import Path

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def _scan_subset_entries(subset_dir: Path) -> list[tuple[str, Path, str, str]]:
    """
    Scan a subset directory and return a list of (utt_id, wav_path, text, spk_key) tuples.

    Args:
        subset_dir: Path to the subset directory (e.g., "LibriTTS/train-clean-100")
    Returns:
        List of tuples containing:
            - utt_id: Unique utterance ID (e.g., "123-456-789")
            - wav_path: Path to the corresponding WAV file
            - text: Transcription text
            - spk_key: Speaker key (e.g., "speaker_chapter") for speaker ID mapping
    """
    entries = []
    for text_path in sorted(subset_dir.rglob("*.normalized.txt")):
        wav_path = text_path.with_suffix("").with_suffix(".wav")
        if not wav_path.is_file():
            continue
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        utt_id = text_path.stem.replace(".normalized", "")
        speaker = text_path.parent.parent.name
        spk_key = speaker
        entries.append((utt_id, wav_path.resolve(), text, spk_key))
    return entries


class LibriTTSBuilder(DatasetBuilder):
    """Prepare LibriTTS manifests and token list for ESPnet3 TTS."""

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        **_kwargs,
    ) -> bool:
        """Check if LibriTTS source data is prepared.
        Args:
            recipe_dir: Recipe root directory (not used in this check).
            **_kwargs: Unused extra options for API compatibility.
        Returns:
            True if the required LibriTTS subsets are present; False otherwise.
        """

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

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Notes:
            This method:
            1. Checks for the presence of required LibriTTS subsets.
            2. If not present, it runs the download script for each missing subset.
            3. Verifies that the required subsets are present after the download attempt.

        """
        dataset_root = Path(recipe_dir).resolve() / _CFG["dataset_path"]

        if not self.is_source_prepared(recipe_dir=recipe_dir):
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
                        f"Check internet connection and disk space."
                    ) from e
        else:
            logger.info("LibriTTS source data is already prepared, skipping download.")

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check if LibriTTS dataset artifacts (manifests) are built.
        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.
        Returns:
            True if all expected manifest files exist; False otherwise.
        """

        recipe_root = Path(recipe_dir).resolve()
        data_dir = recipe_root / _CFG["data_path"]
        manifests_ok = all(
            (data_dir / relpath).is_file()
            for relpath in _CFG["manifest_paths"].values()
        )
        return manifests_ok

    def build(
        self,
        recipe_dir: str | Path,
        **_kwargs,
    ) -> None:
        """Build LibriTTS dataset artifacts (manifests).

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Optional keyword arguments for build customization:

        Returns:
            None.

        Notes:
            Build flow:

        """

        recipe_root = Path(recipe_dir).resolve()
        libritts_root = recipe_root / _CFG["dataset_path"] / "LibriTTS"
        data_dir = recipe_root / _CFG["data_path"]
        data_dir.mkdir(parents=True, exist_ok=True)

        split_entries = {}
        speaker_to_id = {}

        for split, subsets in _CFG["split_subsets"].items():
            entries = []
            for subset in subsets:
                subset_dir = libritts_root / subset
                if not subset_dir.is_dir():
                    raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
                entries.extend(_scan_subset_entries(subset_dir))
            entries = sorted(entries, key=lambda x: x[0])
            split_entries[split] = entries
            for _, _, _, spk_key in entries:
                if spk_key not in speaker_to_id:
                    speaker_to_id[spk_key] = len(speaker_to_id)

        for split, entries in split_entries.items():
            manifest_path = data_dir / _CFG["manifest_paths"][split]
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w", encoding="utf-8") as f:
                for utt_id, wav_path, text, spk_key in entries:
                    sid = speaker_to_id[spk_key]
                    f.write(f"{utt_id}\t{wav_path}\t{text}\t{sid}\n")
