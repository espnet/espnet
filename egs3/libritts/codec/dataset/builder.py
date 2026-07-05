"""LibriTTS dataset builder for the ESPnet3 codec recipe.

The primary manifest stays a plain ``utt_id<TAB>wav_path`` TSV (matching
``LibriTTSCodecDataset``'s parser, which is recipe-specific and intentionally
left alone here) since codec training itself consumes only audio. Reference
labels useful for future evaluation (e.g. ASR-based intelligibility/WER-style
metrics on decoded audio) are collected the same way the reference TTS
builder does (``egs3/libritts/tts/dataset/builder.py`` on ``libritts_vits``)
and written to sidecar ``.text``/``.utt2spk`` files next to each manifest,
rather than appended as extra manifest columns -- that keeps the manifest
format stable for the existing dataset parser while still keeping the labels
around on disk for whenever the `measure` stage needs them.
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


def _scan_subset_entries(subset_dir: Path) -> list[tuple[str, Path, str, str]]:
    """Scan a subset directory and return ``(utt_id, wav_path, text, spk_key)`` tuples.

    Args:
        subset_dir: Path to the subset directory (e.g.,
            ``LibriTTS/train-clean-100``).

    Returns:
        List of ``(utt_id, wav_path, text, spk_key)`` tuples for every
        utterance with both a transcript and a wav file present. ``text`` and
        ``spk_key`` are not part of the audio-only manifest codec training
        reads, but are kept here (mirroring the reference TTS builder) and
        written to sidecar label files by ``build()`` for future evaluation.
    """
    entries = []
    for text_path in sorted(subset_dir.rglob("*.normalized.txt")):
        wav_path = text_path.with_suffix("").with_suffix(".wav")
        if not wav_path.is_file():
            continue
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
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
        spk_key = text_path.parent.parent.name
        entries.append((utt_id, wav_path.resolve(), text, spk_key))
    return entries


def _label_path_for(manifest_relpath: str, label: str) -> str:
    """Derive a sidecar label path (e.g. ``train.text.tsv``) from a manifest path."""
    path = Path(manifest_relpath)
    return str(path.with_name(f"{path.stem}.{label}{path.suffix}"))


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
        """Check if LibriTTS codec manifests and label sidecar files are built."""
        recipe_root = Path(recipe_dir).resolve()
        data_dir = recipe_root / _CFG["data_path"]
        manifests_ok = all(
            (data_dir / relpath).is_file()
            for relpath in _CFG["manifest_paths"].values()
        )
        labels_ok = all(
            (data_dir / _label_path_for(relpath, label)).is_file()
            for relpath in _CFG["manifest_paths"].values()
            for label in ("text", "utt2spk")
        )
        return manifests_ok and labels_ok

    def build(
        self,
        recipe_dir: str | Path,
        **_kwargs,
    ) -> None:
        """Build LibriTTS codec manifests (audio-only) and label sidecar files.

        Raises:
            FileNotFoundError: If a configured subset directory is missing.
        """
        recipe_root = Path(recipe_dir).resolve()
        libritts_root = recipe_root / _CFG["dataset_path"] / "LibriTTS"
        data_dir = recipe_root / _CFG["data_path"]
        data_dir.mkdir(parents=True, exist_ok=True)

        split_entries: dict[str, list[tuple[str, Path, str, str]]] = {}
        speaker_to_id: dict[str, int] = {}

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
            manifest_relpath = _CFG["manifest_paths"][split]
            manifest_path = data_dir / manifest_relpath
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            text_path = data_dir / _label_path_for(manifest_relpath, "text")
            utt2spk_path = data_dir / _label_path_for(manifest_relpath, "utt2spk")

            with manifest_path.open("w", encoding="utf-8") as manifest_f, text_path.open(
                "w", encoding="utf-8"
            ) as text_f, utt2spk_path.open("w", encoding="utf-8") as utt2spk_f:
                for utt_id, wav_path, text, spk_key in entries:
                    manifest_f.write(f"{utt_id}\t{wav_path}\n")
                    text_f.write(f"{utt_id}\t{text}\n")
                    utt2spk_f.write(f"{utt_id}\t{speaker_to_id[spk_key]}\n")
