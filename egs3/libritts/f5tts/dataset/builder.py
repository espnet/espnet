"""LibriTTS dataset builder for the ESPnet3 F5-TTS recipe.

Downloads the LibriTTS subsets listed in ``dataset/config.yaml`` from OpenSLR
and turns them into the TSV manifests consumed by
``egs3.libritts.f5tts.dataset.dataset.LibriTTSDataset``. It also downloads
LibriSpeech ``test-clean`` and the F5-TTS cross-sentence pair list, and
writes the LibriSpeech-PC eval manifest read by ``conf/inference.yaml``.
"""

from __future__ import annotations

import logging
import urllib.error
from importlib import resources
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url, extract_targz

from .librispeech_pc import build_manifest

logger = logging.getLogger(__name__)

# OpenSLR resource 60 is the LibriTTS corpus, 12 is LibriSpeech. Each archive
# extracts to `<corpus>/<subset>/`; LibriSpeech archives are stored under a
# prefixed name because the two corpora publish subsets with identical names
# (e.g. `test-clean.tar.gz`).
_CORPORA: dict[str, tuple[str, str]] = {
    "LibriTTS": ("https://www.openslr.org/resources/60", ""),
    "LibriSpeech": ("https://www.openslr.org/resources/12", "LibriSpeech_"),
}

# Size in bytes of each published `<subset>.tar.gz`. A local archive whose
# size does not match is treated as a partial download and re-fetched.
_ARCHIVE_SIZES: dict[str, dict[str, int]] = {
    "LibriTTS": {
        "dev-clean": 1291469655,
        "test-clean": 1230670113,
        "dev-other": 924804676,
        "test-other": 964502297,
        "train-clean-100": 7723686890,
        "train-clean-360": 27504073644,
        "train-other-500": 44565031479,
    },
    "LibriSpeech": {
        "test-clean": 346663984,
    },
}


def _load_builder_config() -> dict:
    """Return the ``builder`` section of the package's ``config.yaml``."""
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
    corpus: str = "LibriTTS",
    remove_archive: bool = False,
) -> None:
    """Download and extract one ``corpus`` subset into ``dataset_root``.

    Idempotent: a subset whose ``<corpus>/<subset>/.complete`` marker already
    exists is skipped, and an already downloaded archive of the expected size
    is reused instead of being fetched again.
    """
    if corpus not in _CORPORA:
        raise ValueError(
            f"Unknown corpus '{corpus}'. Expected one of {sorted(_CORPORA)}"
        )
    url_base, archive_prefix = _CORPORA[corpus]
    sizes = _ARCHIVE_SIZES[corpus]
    if subset not in sizes:
        raise ValueError(
            f"Unknown {corpus} subset '{subset}'. Expected one of {sorted(sizes)}"
        )

    marker = dataset_root / corpus / subset / ".complete"
    if marker.is_file():
        logger.info("%s subset %s already downloaded, skipping.", corpus, subset)
        return

    archive_path = dataset_root / f"{archive_prefix}{subset}.tar.gz"
    expected_size = sizes[subset]
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
        logger.info("Downloading %s subset: %s", corpus, subset)
        download_url(
            f"{url_base}/{subset}.tar.gz",
            archive_path,
            logger=logger,
        )

    extract_targz(archive_path, dataset_root, logger=logger)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    logger.info("Successfully downloaded and extracted %s %s", corpus, subset)

    if remove_archive:
        archive_path.unlink(missing_ok=True)
        logger.info("Removed archive %s", archive_path)


def _download_pair_list(url: str, dest: Path) -> None:
    """Download the LibriSpeech-PC pair list to ``dest``.

    The list is a small text file with no published size to check against,
    so it is downloaded to a temporary sibling and renamed: an interrupted
    transfer never leaves a truncated file that later runs would treat as
    complete.

    Raises:
        RuntimeError: If the download fails.
    """
    tmp_path = dest.with_name(dest.name + ".tmp")
    logger.info("Downloading LibriSpeech-PC pair list from %s", url)
    try:
        download_url(url, tmp_path, logger=logger)
        tmp_path.replace(dest)
    except (urllib.error.URLError, OSError) as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download the LibriSpeech-PC pair list from {url}. "
            f"Check your internet connection, or download the file manually "
            f"and place it at {dest}."
        ) from e


def _scan_subset_entries(subset_dir: Path) -> list[tuple[str, Path, str, str]]:
    """
    Scan a subset directory and return a list of
    (utt_id, wav_path, text, spk_key) tuples.

    Args:
        subset_dir: Path to the subset directory (e.g., "LibriTTS/train-clean-100")
    Returns:
        List of tuples containing:
            - utt_id: Unique utterance ID (e.g., "123_456_789_000")
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


def _librispeech_pc_paths(recipe_root: Path) -> tuple[Path, Path, Path]:
    """Resolve the three LibriSpeech-PC paths from the builder config.

    Args:
        recipe_root: Resolved recipe root directory.

    Returns:
        Tuple of ``(test_clean_root, lst_path, manifest_path)``. The first two
        live under ``builder.dataset_path``, the last under
        ``builder.data_path``.
    """
    cfg = _CFG["librispeech_pc"]
    dataset_root = recipe_root / _CFG["dataset_path"]
    data_dir = recipe_root / _CFG["data_path"]
    return (
        dataset_root / cfg["test_clean_path"],
        dataset_root / cfg["lst_path"],
        data_dir / cfg["manifest_path"],
    )


class LibriTTSBuilder(DatasetBuilder):
    """Prepare LibriTTS and LibriSpeech-PC manifests for the F5-TTS recipe."""

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        **_kwargs,
    ) -> bool:
        """Check if the source corpora are prepared.

        A subset counts as prepared only when ``prepare_source`` finished
        extracting it, which is recorded by the ``<corpus>/<subset>/.complete``
        marker. Testing the directory alone would accept the partial tree an
        interrupted extraction leaves behind, and ``build`` would then write
        manifests from incomplete source data.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            True if every required LibriTTS subset and LibriSpeech test-clean
            carry their ``.complete`` marker and the LibriSpeech-PC pair list
            is present; False otherwise.

        Note:
            When a corpus was staged by hand instead of by ``prepare_source``,
            create the markers so this check passes:
            ``touch <dataset_path>/<corpus>/<subset>/.complete`` for each
            configured subset. Without them this returns False and
            ``prepare_source`` re-downloads the archives.

        Examples:
            ```python
            builder = LibriTTSBuilder()
            if not builder.is_source_prepared(recipe_dir="egs3/libritts/f5tts"):
                builder.prepare_source(recipe_dir="egs3/libritts/f5tts")
            ```
        """
        recipe_root = Path(recipe_dir).resolve()
        dataset_root = recipe_root / _CFG["dataset_path"]
        _, lst_path, _ = _librispeech_pc_paths(recipe_root)
        libritts_ready = all(
            (dataset_root / "LibriTTS" / subset / ".complete").is_file()
            for subset in _required_subsets()
        )
        librispeech_subset = _CFG["librispeech_pc"]["subset"]
        librispeech_ready = (
            dataset_root / "LibriSpeech" / librispeech_subset / ".complete"
        ).is_file()
        return libritts_ready and librispeech_ready and lst_path.is_file()

    def prepare_source(
        self,
        recipe_dir: str | Path,
        remove_archive: bool = False,
        **_kwargs,
    ) -> None:
        """Download the corpora required by this recipe.

        Each LibriTTS subset listed under ``builder.split_subsets`` in
        ``dataset/config.yaml`` is fetched from OpenSLR into
        ``<recipe_dir>/<builder.dataset_path>`` and extracted there, then
        LibriSpeech ``test-clean`` (the audio of the default eval set) and the
        F5-TTS cross-sentence pair list. A ``<corpus>/<subset>/.complete``
        marker makes each download idempotent, so an interrupted run can
        simply be restarted.

        Args:
            recipe_dir: Recipe root directory.
            remove_archive: Delete each ``.tar.gz`` after a successful
                extraction. Useful when disk space is tight; the default keeps
                the archives so a re-run does not download them again.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            ValueError: If a configured subset is not a known LibriTTS subset.
            URLError: If an archive download fails.
            RuntimeError: If the pair list download fails.

        Examples:
            Called by the ``create_dataset`` stage, but it can also be driven
            directly:
            ```python
            from egs3.libritts.f5tts.dataset.builder import LibriTTSBuilder

            builder = LibriTTSBuilder()
            builder.prepare_source(recipe_dir="egs3/libritts/f5tts")
            ```

            The full recipe download is ~80 GB. To keep only the extracted
            audio:
            ```python
            builder.prepare_source(
                recipe_dir="egs3/libritts/f5tts",
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
            logger.info("Source data is already prepared, skipping download.")
            return

        recipe_root = Path(recipe_dir).resolve()
        dataset_root = recipe_root / _CFG["dataset_path"]
        dataset_root.mkdir(parents=True, exist_ok=True)
        for subset in _required_subsets():
            _download_subset(dataset_root, subset, remove_archive=remove_archive)

        lspc_cfg = _CFG["librispeech_pc"]
        _download_subset(
            dataset_root,
            lspc_cfg["subset"],
            corpus="LibriSpeech",
            remove_archive=remove_archive,
        )
        _, lst_path, _ = _librispeech_pc_paths(recipe_root)
        if lst_path.is_file():
            logger.info("LibriSpeech-PC pair list already downloaded, skipping.")
        else:
            _download_pair_list(lspc_cfg["lst_url"], lst_path)

    def is_libritts_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check only the LibriTTS split manifests, which training reads.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            True if the LibriTTS split manifests exist; False otherwise.

        Note:
            Deliberately narrower than :meth:`is_built`. ``LibriTTSDataset``
            guards on this one, so training is not blocked by a missing
            LibriSpeech-PC eval manifest it never reads.
        """
        data_dir = Path(recipe_dir).resolve() / _CFG["data_path"]
        return all(
            (data_dir / relpath).is_file()
            for relpath in _CFG["manifest_paths"].values()
        )

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check if the dataset artifacts (manifests) are built.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            True if the LibriTTS split manifests and the LibriSpeech-PC eval
            manifest all exist; False otherwise.

        Note:
            The LibriSpeech-PC manifest is part of this check because
            ``conf/inference.yaml``, the default eval config, reads it. Use
            :meth:`is_libritts_built` for the training-only subset.

        Examples:
            ```python
            builder = LibriTTSBuilder()
            if not builder.is_built(recipe_dir="egs3/libritts/f5tts"):
                builder.build(recipe_dir="egs3/libritts/f5tts")
            ```
        """
        recipe_root = Path(recipe_dir).resolve()
        _, _, lspc_manifest = _librispeech_pc_paths(recipe_root)
        return self.is_libritts_built(recipe_dir=recipe_root) and (
            lspc_manifest.is_file()
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
        The LibriSpeech-PC eval manifest is then written from the pair list
        (see :func:`egs3.libritts.f5tts.dataset.librispeech_pc.build_manifest`).
        This method performs no network I/O: everything it reads is fetched by
        ``prepare_source``.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            None. Manifests are written under
            ``<recipe_dir>/<builder.data_path>/<builder.manifest_paths[split]>``
            and ``<builder.librispeech_pc.manifest_path>``.

        Raises:
            FileNotFoundError: If a configured subset directory, the
                LibriSpeech test-clean tree or the pair list is missing, i.e.
                ``prepare_source`` has not run successfully.

        Examples:
            ```python
            from egs3.libritts.f5tts.dataset.builder import LibriTTSBuilder

            builder = LibriTTSBuilder()
            builder.prepare_source(recipe_dir="egs3/libritts/f5tts")
            builder.build(recipe_dir="egs3/libritts/f5tts")
            ```

            Each split manifest row is four tab-separated fields, e.g.:
            ```text
            1089_134691_000004_000001
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

        test_clean_root, lst_path, lspc_manifest = _librispeech_pc_paths(recipe_root)
        for path, what in (
            (lst_path, "LibriSpeech-PC pair list"),
            (test_clean_root, "LibriSpeech test-clean tree"),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {what}: {path}. Run the create_dataset stage so "
                    f"prepare_source() downloads it."
                )
        n_rows = build_manifest(lst_path, test_clean_root, lspc_manifest)
        logger.info("Wrote %d LibriSpeech-PC rows to %s", n_rows, lspc_manifest)
