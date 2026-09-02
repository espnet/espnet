"""LibriTTS dataset builder for the ESPnet3 codec recipe.

Downloads the LibriTTS subsets listed in ``dataset/config.yaml`` from OpenSLR
and turns them into the TSV manifests consumed by
``egs3.libritts.codec.dataset.dataset.LibriTTSCodecDataset``.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url, extract_targz

logger = logging.getLogger(__name__)

# OpenSLR resource 60 is the LibriTTS corpus.
_OPENSLR_URL_BASE = "https://www.openslr.org/resources/60"

# Size in bytes of each published `<subset>.tar.gz`. A local archive whose
# size does not match is treated as a partial download and re-fetched.
_ARCHIVE_SIZES: dict[str, int] = {
    "dev-clean": 1291469655,
    "test-clean": 1230670113,
    "dev-other": 924804676,
    "test-other": 964502297,
    "train-clean-100": 7723686890,
    "train-clean-360": 27504073644,
    "train-other-500": 44565031479,
}


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def _required_subsets() -> list[str]:
    """Return every LibriTTS subset referenced by the split definitions."""
    required: list[str] = []
    for subsets in _CFG["split_subsets"].values():
        required.extend(subsets)
    return required


def _download_subset(
    dataset_root: Path,
    subset: str,
    remove_archive: bool = False,
) -> None:
    """Download and extract one LibriTTS subset into ``dataset_root``.

    Idempotent: a subset whose ``LibriTTS/<subset>/.complete`` marker already
    exists is skipped, and an already downloaded archive of the expected size
    is reused instead of being fetched again.
    """
    if subset not in _ARCHIVE_SIZES:
        raise ValueError(
            f"Unknown LibriTTS subset '{subset}'. "
            f"Expected one of {sorted(_ARCHIVE_SIZES)}"
        )

    marker = dataset_root / "LibriTTS" / subset / ".complete"
    if marker.is_file():
        logger.info("Subset %s already downloaded, skipping.", subset)
        return

    archive_path = dataset_root / f"{subset}.tar.gz"
    expected_size = _ARCHIVE_SIZES[subset]
    if archive_path.is_file():
        actual_size = archive_path.stat().st_size
        if actual_size == expected_size:
            logger.info("Reusing existing archive %s", archive_path)
        else:
            logger.warning(
                "Removing incomplete archive %s (%d bytes, expected %d)",
                archive_path,
                actual_size,
                expected_size,
            )
            archive_path.unlink()

    if not archive_path.is_file():
        logger.info("Downloading LibriTTS subset: %s", subset)
        download_url(
            f"{_OPENSLR_URL_BASE}/{subset}.tar.gz",
            archive_path,
            logger=logger,
        )

    extract_targz(archive_path, dataset_root, logger=logger)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    logger.info("Successfully downloaded and extracted %s", subset)

    if remove_archive:
        archive_path.unlink(missing_ok=True)
        logger.info("Removed archive %s", archive_path)


def _scan_subset_entries(subset_dir: Path) -> list[tuple[str, Path, str, str]]:
    """
    Scan a subset directory and return a list of
    (utt_id, wav_path, text, spk_key) tuples.

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
        utt_id = text_path.stem.replace(".normalized", "").replace("_", "-")
        speaker = text_path.parent.parent.name
        spk_key = speaker
        entries.append((utt_id, wav_path.resolve(), text, spk_key))
    return entries


class LibriTTSBuilder(DatasetBuilder):
    """Prepare LibriTTS manifests for the ESPnet3 codec recipe."""

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        **_kwargs,
    ) -> bool:
        """Check if LibriTTS source data is prepared.

        A subset counts as prepared only when ``prepare_source`` finished
        extracting it, which is recorded by the ``LibriTTS/<subset>/.complete``
        marker. Testing the directory alone would accept the partial tree an
        interrupted extraction leaves behind, and ``build`` would then write
        manifests from incomplete source data.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            True if every required LibriTTS subset carries its ``.complete``
            marker; False otherwise.

        Note:
            When the corpus was staged by hand instead of by
            ``prepare_source``, create the markers so this check passes:
            ``touch <dataset_path>/LibriTTS/<subset>/.complete`` for each
            configured subset. Without them this returns False and
            ``prepare_source`` re-downloads the archives.

        Examples:
            ```python
            builder = LibriTTSBuilder()
            if not builder.is_source_prepared(recipe_dir="egs3/libritts/codec"):
                builder.prepare_source(recipe_dir="egs3/libritts/codec")
            ```
        """
        recipe_root = Path(recipe_dir).resolve()
        libritts_root = recipe_root / _CFG["dataset_path"] / "LibriTTS"
        return all(
            (libritts_root / subset / ".complete").is_file()
            for subset in _required_subsets()
        )

    def prepare_source(
        self,
        recipe_dir: str | Path,
        remove_archive: bool = False,
        **_kwargs,
    ) -> None:
        """Download the LibriTTS subsets required by this recipe.

        Each subset listed under ``builder.split_subsets`` in
        ``dataset/config.yaml`` is fetched from OpenSLR into
        ``<recipe_dir>/<builder.dataset_path>`` and extracted there. A
        ``LibriTTS/<subset>/.complete`` marker makes the download idempotent,
        so an interrupted run can simply be restarted.

        Args:
            recipe_dir: Recipe root directory.
            remove_archive: Delete each ``<subset>.tar.gz`` after a successful
                extraction. Useful when disk space is tight; the default keeps
                the archives so a re-run does not download them again.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            ValueError: If a configured subset is not a known LibriTTS subset.
            URLError: If a download fails.

        Examples:
            Called by the ``create_dataset`` stage, but it can also be driven
            directly:
            ```python
            from egs3.libritts.codec.dataset.builder import LibriTTSBuilder

            builder = LibriTTSBuilder()
            builder.prepare_source(recipe_dir="egs3/libritts/codec")
            ```

            The full recipe download is ~80 GB. To keep only the extracted
            audio:
            ```python
            builder.prepare_source(
                recipe_dir="egs3/libritts/codec",
                remove_archive=True,
            )
            ```

            The `create_dataset` stage forwards every key under
            `create_dataset:` in the training config as a builder kwarg, so
            the same option is reachable from yaml:
            ```yaml
            create_dataset:
              recipe_dir: ${recipe_dir}
              remove_archive: true
            ```
        """
        if self.is_source_prepared(recipe_dir=recipe_dir):
            logger.info("LibriTTS source data is already prepared, skipping download.")
            return

        dataset_root = Path(recipe_dir).resolve() / _CFG["dataset_path"]
        dataset_root.mkdir(parents=True, exist_ok=True)
        for subset in _required_subsets():
            _download_subset(dataset_root, subset, remove_archive=remove_archive)

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check if LibriTTS dataset artifacts (manifests) are built.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            True if all expected manifest files exist; False otherwise.

        Examples:
            ```python
            builder = LibriTTSBuilder()
            if not builder.is_built(recipe_dir="egs3/libritts/codec"):
                builder.build(recipe_dir="egs3/libritts/codec")
            ```
        """
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
        """Write one ``utt_id<TAB>wav_path<TAB>text<TAB>sid`` manifest per split.

        Every subset of a split is scanned for LibriTTS ``*.normalized.txt``
        files and their sibling ``*.wav``. Speaker IDs are assigned across all
        splits at once, so the same speaker gets the same integer everywhere.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            None. Manifests are written under
            ``<recipe_dir>/<builder.data_path>/<builder.manifest_paths[split]>``.

        Raises:
            FileNotFoundError: If a configured subset directory is missing,
                i.e. ``prepare_source`` has not run successfully.

        Examples:
            ```python
            from egs3.libritts.codec.dataset.builder import LibriTTSBuilder

            builder = LibriTTSBuilder()
            builder.prepare_source(recipe_dir="egs3/libritts/codec")
            builder.build(recipe_dir="egs3/libritts/codec")
            ```

            Each manifest row is four tab-separated fields, e.g.:
            ```text
            1089-134691-000004-000001
            /abs/path/1089_134691_000004_000001.wav
            He hoped there would be stew for dinner.
            0
            ```
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
