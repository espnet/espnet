"""LibriTTS dataset builder for the ESPnet3 espnet2 GAN-TTS recipe.

Downloads the LibriTTS subsets listed in ``dataset/config.yaml`` from OpenSLR,
resamples the audio once to the recipe sampling rate (22.05 kHz, as the
espnet2 LibriTTS VITS recipe does with ``format_wav_scp --fs 22050``) and
turns them into the TSV manifests consumed by
``egs3.libritts.esp2_gan_tts.dataset.dataset.LibriTTSDataset``.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from importlib import resources
from pathlib import Path

import numpy as np
import soundfile as sf

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url, extract_targz

logger = logging.getLogger(__name__)

# OpenSLR resource 60 is the LibriTTS corpus.
_OPENSLR_URL_BASE = "https://www.openslr.org/resources/60"

# Size in bytes of each published ``<subset>.tar.gz``. A local archive whose
# size does not match is treated as a partial download and re-fetched.
_ARCHIVE_SIZES: dict[str, int] = {
    "dev-clean": 1291469655,
    "dev-other": 924804676,
    "test-clean": 1230670113,
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


def _resample_wav(src: Path, dst: Path, fs: int) -> bool:
    """Write ``src`` at ``fs`` Hz to ``dst`` as PCM_16; return False if it existed.

    Mirrors what espnet2's ``format_wav_scp.py --fs`` does to the dumped
    audio: resample when the file rate differs, write 16-bit PCM wav. espnet2
    resamples with resampy (an optional ``recipe`` extra); this uses
    ``torchaudio.functional.resample`` (windowed sinc), which is a core
    dependency. The file is written to a temporary name and renamed, so an
    interrupted build never leaves a truncated wav behind, and an existing
    ``dst`` is kept so a re-run resumes where it stopped.
    """
    if dst.is_file():
        return False
    wav, rate = sf.read(str(src), dtype="float32", always_2d=False)
    if rate != fs:
        import torch
        import torchaudio

        torch.set_num_threads(1)
        tensor = torch.from_numpy(np.ascontiguousarray(wav))
        wav = torchaudio.functional.resample(tensor, rate, fs).numpy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(dst.name + ".tmp")
    sf.write(str(tmp_path), wav, fs, subtype="PCM_16", format="WAV")
    os.replace(tmp_path, dst)
    return True


def _resample_job(job: tuple[Path, Path, int]) -> bool:
    """Unpack one ``(src, dst, fs)`` job for the process pool."""
    return _resample_wav(*job)


def _resample_all(jobs: list[tuple[Path, Path, int]], num_workers: int) -> int:
    """Resample every ``(src, dst, fs)`` job, in a process pool when asked.

    Returns the number of files actually written (skipped files excluded).
    A ``spawn`` context is used so the torch state of the driver process is
    never inherited by forked workers.
    """
    if num_workers <= 1:
        return sum(_resample_job(job) for job in jobs)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as pool:
        return sum(pool.map(_resample_job, jobs, chunksize=64))


def _scan_subset_entries(subset_dir: Path) -> list[tuple[str, Path, str, str]]:
    """Scan a subset directory and return ``(utt_id, wav_path, text, spk_key)`` tuples.

    Args:
        subset_dir: Path to the subset directory (e.g.,
            ``LibriTTS/train-clean-100``).

    Returns:
        List of tuples containing:
            - utt_id: Unique utterance ID (e.g., ``123-456-789``).
            - wav_path: Path to the corresponding WAV file.
            - text: Transcription text.
            - spk_key: Speaker key (e.g., ``speaker_chapter``) for speaker
              ID mapping.
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
    """Prepare LibriTTS manifests for the ESPnet3 TTS recipe.

    ``dataset/config.yaml`` drives both halves of the builder API:
    ``prepare_source`` fetches every subset listed under ``split_subsets``
    from OpenSLR, and ``build`` resamples the audio to ``builder.fs`` under
    ``<data_path>/<audio_path>`` and turns it into one TSV manifest per split
    holding ``utt_id``, ``wav_path``, ``text`` and a corpus-wide speaker ID.

    Examples:
        ```python
        builder = LibriTTSBuilder()
        builder.prepare_source(recipe_dir="egs3/libritts/esp2_gan_tts")
        builder.build(recipe_dir="egs3/libritts/esp2_gan_tts")
        # -> egs3/libritts/esp2_gan_tts/data/manifest/{train,valid,test}.tsv
        ```
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        **_kwargs,
    ) -> bool:
        """Check if LibriTTS source data is prepared.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            ``True`` if every required LibriTTS subset carries the
            ``.complete`` marker that ``prepare_source`` writes after a
            successful extraction; ``False`` otherwise.

        Notes:
            A bare subset directory is not enough: an interrupted extraction
            leaves one behind, and building manifests from it would silently
            drop utterances. If you place a LibriTTS copy by hand instead of
            letting ``prepare_source`` download it, create the marker
            yourself (``touch downloads/LibriTTS/<subset>/.complete``) or
            the next ``create_dataset`` run will re-download the subset.

        Examples:
            ```python
            LibriTTSBuilder().is_source_prepared(
                recipe_dir="egs3/libritts/esp2_gan_tts"
            )
            # -> False until downloads/LibriTTS/<subset>/.complete exists
            #    for every subset
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
        so an interrupted run can simply be restarted, and a partially
        downloaded archive is detected by size and re-fetched.

        Args:
            recipe_dir: Recipe root directory.
            remove_archive: Delete each ``<subset>.tar.gz`` after a successful
                extraction. Useful when disk space is tight; the default keeps
                the archives so a re-run does not download them again.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            None.

        Raises:
            ValueError: If a configured subset is not a known LibriTTS subset.
            URLError: If a download fails.

        Examples:
            Called by the ``create_dataset`` stage, but it can also be driven
            directly:
            ```python
            from egs3.libritts.esp2_gan_tts.dataset.builder import LibriTTSBuilder

            builder = LibriTTSBuilder()
            builder.prepare_source(recipe_dir="egs3/libritts/esp2_gan_tts")
            ```

            The full recipe download is ~80 GB. To keep only the extracted
            audio:
            ```python
            builder.prepare_source(
                recipe_dir="egs3/libritts/esp2_gan_tts",
                remove_archive=True,
            )
            ```

            The ``create_dataset`` stage forwards every key under
            ``create_dataset:`` in the training config as a builder kwarg, so
            the same option is reachable from yaml:
            ```yaml
            create_dataset:
              recipe_dir: ${recipe_dir}
              remove_archive: true
            ```
        """
        dataset_root = Path(recipe_dir).resolve() / _CFG["dataset_path"]

        if self.is_source_prepared(recipe_dir=recipe_dir):
            logger.info("LibriTTS source data is already prepared, skipping download.")
            return

        dataset_root.mkdir(parents=True, exist_ok=True)
        for subset in _required_subsets():
            _download_subset(dataset_root, subset, remove_archive=remove_archive)

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check if the resampled audio and the manifests are built.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            ``True`` if every manifest exists and the resampled-audio tree
            carries the ``.complete`` marker ``build`` writes last; ``False``
            otherwise.

        Notes:
            Manifests alone are not enough: a checkout built before the
            recipe resampled its audio has manifests that point at the 24 kHz
            originals. Such a tree has no ``<audio_path>/.complete`` marker,
            so this returns ``False`` and ``create_dataset`` rebuilds it.

        Examples:
            ```python
            LibriTTSBuilder().is_built(recipe_dir="egs3/libritts/esp2_gan_tts")
            # -> True once data/wav/.complete and
            #    data/manifest/{train,valid,test}.tsv all exist
            ```
        """
        recipe_root = Path(recipe_dir).resolve()
        data_dir = recipe_root / _CFG["data_path"]
        manifests_ok = all(
            (data_dir / relpath).is_file()
            for relpath in _CFG["manifest_paths"].values()
        )
        audio_ok = (data_dir / _CFG["audio_path"] / ".complete").is_file()
        return manifests_ok and audio_ok

    def build(
        self,
        recipe_dir: str | Path,
        fs: int | None = None,
        num_workers: int = 1,
        **_kwargs,
    ) -> None:
        """Build the task-ready audio (resampled copy) and the manifests.

        Args:
            recipe_dir: Recipe root directory.
            fs: Sampling rate of the written audio. Defaults to
                ``builder.fs`` in ``dataset/config.yaml`` (22050, the rate the
                espnet2 LibriTTS VITS recipe trains at). Must equal the
                ``fs`` used by the preprocessor and the model config.
            num_workers: Resampling processes. ``1`` runs in-process; the
                full corpus is ~375k files, so a larger value is advisable on
                a multi-core node. Reachable from yaml as
                ``create_dataset.num_workers``.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            None.

        Raises:
            FileNotFoundError: If a configured subset directory is missing.

        Notes:
            Build flow:
            1. Scan each split's subset directories for utterance entries.
            2. Resample every wav to ``fs`` under
               ``<data_path>/<audio_path>/<subset>/<speaker>/<chapter>/``
               as PCM_16, skipping files already written, each one via a
               temporary name and a rename.
            3. Assign incrementing speaker IDs in first-seen order.
            4. Write TSV manifests for ``train``, ``valid``, ``test`` that
               point at the resampled files.
            5. Touch ``<audio_path>/.complete`` so ``is_built`` passes.

            Speaker IDs are assigned over all splits at once, so a speaker
            appearing in more than one split keeps a single ID. A PCM_16 copy
            of the full corpus at 22.05 kHz needs roughly 90 GB on top of the
            originals; drop ``train-other-500`` from ``split_subsets`` or
            pass ``remove_archive`` to ``prepare_source`` if disk is tight.

        Examples:
            ```python
            LibriTTSBuilder().build(recipe_dir="egs3/libritts/esp2_gan_tts")
            ```
            writes one tab-separated row per utterance, as
            ``utt_id, wav_path, text, speaker_id``, with ``wav_path`` under the
            resampled tree, e.g. for utterance ``1272_128104_000001_000000``:
            ```text
            <utt_id>\tdata/wav/dev-clean/1272/128104/<utt_id>.wav\tIt is a ...\t0
            ```
            From yaml, the same options are forwarded by the stage:
            ```yaml
            create_dataset:
              recipe_dir: ${recipe_dir}
              num_workers: 16
            ```
        """
        fs = int(_CFG["fs"] if fs is None else fs)
        recipe_root = Path(recipe_dir).resolve()
        libritts_root = recipe_root / _CFG["dataset_path"] / "LibriTTS"
        data_dir = recipe_root / _CFG["data_path"]
        audio_root = data_dir / _CFG["audio_path"]
        data_dir.mkdir(parents=True, exist_ok=True)

        split_entries = {}
        speaker_to_id = {}
        jobs: list[tuple[Path, Path, int]] = []

        for split, subsets in _CFG["split_subsets"].items():
            entries = []
            for subset in subsets:
                subset_dir = libritts_root / subset
                if not subset_dir.is_dir():
                    raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
                scanned = _scan_subset_entries(subset_dir)
                for utt_id, src_path, text, spk_key in scanned:
                    relative = src_path.relative_to(libritts_root.resolve())
                    dst_path = audio_root / relative
                    jobs.append((src_path, dst_path, fs))
                    entries.append((utt_id, dst_path, text, spk_key))
            entries = sorted(entries, key=lambda x: x[0])
            split_entries[split] = entries
            for _, _, _, spk_key in entries:
                if spk_key not in speaker_to_id:
                    speaker_to_id[spk_key] = len(speaker_to_id)

        logger.info(
            "Resampling %d files to %d Hz under %s (num_workers=%d)",
            len(jobs),
            fs,
            audio_root,
            num_workers,
        )
        n_written = _resample_all(jobs, num_workers)
        logger.info(
            "Resampled %d files (%d already present)",
            n_written,
            len(jobs) - n_written,
        )

        for split, entries in split_entries.items():
            manifest_path = data_dir / _CFG["manifest_paths"][split]
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                for utt_id, wav_path, text, spk_key in entries:
                    sid = speaker_to_id[spk_key]
                    f.write(f"{utt_id}\t{wav_path}\t{text}\t{sid}\n")
            os.replace(tmp_path, manifest_path)

        audio_root.mkdir(parents=True, exist_ok=True)
        (audio_root / ".complete").touch()
