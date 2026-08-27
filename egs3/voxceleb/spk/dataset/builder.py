"""VoxCeleb 1 & 2 dataset builder for speaker verification."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf
from tqdm import tqdm

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.parallel.parallel import get_client
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

AUDIO_SUFFIX = ".wav"
MANIFESTS = ("wav.scp", "utt2spk", "spk2utt")


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def _env_path(env_var: str) -> Path | None:
    """Return the directory named by an environment variable, if it is set."""
    value = os.environ.get(str(env_var))
    return Path(value) if value else None


def iter_source_candidates(
    recipe_root: Path,
    source_dir: str | Path | None,
) -> Iterable[Path]:
    """Yield candidate directories that may hold the VoxCeleb audio."""
    if source_dir is not None:
        yield Path(source_dir)
    env_root = _env_path(_CFG["source_env_var"])
    if env_root is not None:
        yield env_root
    yield recipe_root / _CFG["download_path"]


def resolve_source_root(
    recipe_root: Path,
    source_dir: str | Path | None = None,
) -> Path:
    """Resolve the VoxCeleb root that holds the configured audio directories.

    Args:
        recipe_root: Recipe root directory.
        source_dir: Optional override pointing at the VoxCeleb root.

    Returns:
        Path to the resolved VoxCeleb root.

    Raises:
        FileNotFoundError: If no candidate directory contains the audio trees
            listed under ``builder.sources``.
    """
    checked: list[str] = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        checked.append(str(candidate))
        if all((candidate / str(rel)).is_dir() for rel in _CFG["sources"].values()):
            return candidate

    expected = ", ".join(str(rel) for rel in _CFG["sources"].values())
    raise FileNotFoundError(
        "VoxCeleb source not found. Checked these locations:\n"
        + "\n".join(f"  - {path}" for path in checked)
        + f"\nEach location must contain: {expected}\n"
        + f"Set the {_CFG['source_env_var']} environment variable to the "
        + "VoxCeleb root, or pass `source_dir` in `create_dataset`."
    )


def audio_suffixes() -> tuple[str, ...]:
    """Return the file extensions the builder indexes, WAV first."""
    return (AUDIO_SUFFIX, *(str(s) for s in _CFG["convert_suffixes"]))


def convert_audio(job: tuple[str, str], sample_rate: int, channels: int) -> str:
    """Decode one source file into a WAV of the recipe's sample rate.

    This runs on a Dask worker, so it takes and returns plain strings and shells
    out to ffmpeg instead of importing an audio backend. The output is written
    to a temporary file and renamed into place, so an interrupted run never
    leaves a truncated WAV that a later run would mistake for a finished one.

    Args:
        job: ``(source, target)`` pair of paths.
        sample_rate: Output sample rate in Hz.
        channels: Number of output channels.

    Returns:
        The target path that was written.

    Raises:
        RuntimeError: If ffmpeg fails on the source file.
    """
    source, target = Path(job[0]), Path(job[1])
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")

    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            str(partial),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg failed to convert {source}:\n{result.stderr.strip()}"
        )

    partial.replace(target)
    return str(target)


def scan_split(split_dir: Path) -> list[tuple[str, str, Path]]:
    """Index one VoxCeleb audio tree as ``(utt_id, speaker, path)`` entries.

    VoxCeleb is laid out as ``<speaker>/<video>/<utterance>.<ext>`` and the
    utterance ID is that relative path without its suffix, which is also how
    the official trial lists refer to utterances. The WAV of VoxCeleb1 and the
    AAC of VoxCeleb2 are indexed alike; :func:`plan_conversions` decides which
    of them still have to be decoded.

    Args:
        split_dir: Directory holding one VoxCeleb split.

    Returns:
        Entries sorted by utterance ID.

    Raises:
        RuntimeError: If the directory contains no audio the builder can read.
    """
    entries = []
    for suffix in audio_suffixes():
        for path in split_dir.rglob(f"*{suffix}"):
            speaker, video, utterance = path.parts[-3:]
            utt_id = f"{speaker}/{video}/{utterance[: -len(suffix)]}"
            entries.append((utt_id, speaker, path.resolve()))

    if not entries:
        raise RuntimeError(
            f"No audio found under: {split_dir}. Expected files ending in "
            f"{', '.join(audio_suffixes())}."
        )
    return sorted(entries)


def plan_conversions(
    convert_root: Path,
    split: str,
    entries: list[tuple[str, str, Path]],
) -> tuple[list[tuple[str, str, Path]], list[tuple[str, str]]]:
    """Point non-WAV entries at their converted paths and list the work left.

    Targets that already exist are left out of the returned jobs, so a
    conversion interrupted halfway through resumes instead of restarting.

    Args:
        convert_root: Directory the converted audio is written under.
        split: Split name, used as the first level under ``convert_root``.
        entries: Entries as returned by :func:`scan_split`.

    Returns:
        ``(entries, jobs)`` where every entry path now ends in ``.wav``, and
        ``jobs`` holds the ``(source, target)`` pairs still to convert.
    """
    resolved: list[tuple[str, str, Path]] = []
    jobs: list[tuple[str, str]] = []
    for utt_id, speaker, path in entries:
        if path.suffix == AUDIO_SUFFIX:
            resolved.append((utt_id, speaker, path))
            continue
        target = convert_root / split / f"{utt_id}{AUDIO_SUFFIX}"
        resolved.append((utt_id, speaker, target))
        if not target.is_file():
            jobs.append((str(path), str(target)))
    return resolved, jobs


def run_conversions(jobs: list[tuple[str, str]], n_workers: int | None = None) -> None:
    """Decode every planned file to WAV through ``espnet3.parallel``.

    Args:
        jobs: ``(source, target)`` pairs as returned by :func:`plan_conversions`.
        n_workers: Overrides ``builder.parallel.n_workers`` from the config.

    Raises:
        RuntimeError: If ffmpeg is not on ``PATH``, or fails on any file.
    """
    if not jobs:
        return

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"{len(jobs)} VoxCeleb file(s) are not WAV and ffmpeg is not on "
            "PATH to convert them. Install ffmpeg, or convert the corpus "
            f"yourself and point {_CFG['source_env_var']} at the result."
        )

    config = OmegaConf.create(OmegaConf.to_container(_CFG["parallel"], resolve=True))
    if n_workers is not None:
        config.n_workers = int(n_workers)

    sample_rate = int(_CFG["sample_rate"])
    channels = int(_CFG["channels"])
    logger.info(
        "Converting %d file(s) to %d Hz %d-channel WAV with %d worker(s)",
        len(jobs),
        sample_rate,
        channels,
        config.n_workers,
    )

    with get_client(config) as client:
        # Imported here, and after `get_client`, so that the module still
        # imports without Dask and a missing Dask still raises the install
        # hint from `espnet3.parallel` rather than a bare ImportError.
        from dask.distributed import as_completed

        futures = client.map(
            convert_audio, jobs, sample_rate=sample_rate, channels=channels
        )
        try:
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="ffmpeg"
            ):
                future.result()
        except Exception:
            client.cancel(futures)
            raise


def write_spk2utt(split_dir: Path, entries: list[tuple[str, str, Path]]) -> int:
    """Write the ``spk2utt`` of one split or label space.

    Args:
        split_dir: Directory to write into. Created if it does not exist.
        entries: Entries as returned by :func:`scan_split`.

    Returns:
        The number of speakers written, which is the ``model.spk_num`` a
        training config must declare when it trains on these entries.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    spk2utt: dict[str, list[str]] = {}
    for utt_id, speaker, _path in entries:
        spk2utt.setdefault(speaker, []).append(utt_id)

    with (split_dir / "spk2utt").open("w", encoding="utf-8") as f:
        for speaker in sorted(spk2utt):
            f.write(f"{speaker} {' '.join(spk2utt[speaker])}\n")
    return len(spk2utt)


def write_manifests(split_dir: Path, entries: list[tuple[str, str, Path]]) -> None:
    """Write ``wav.scp``, ``utt2spk``, and ``spk2utt`` for one split."""
    split_dir.mkdir(parents=True, exist_ok=True)
    with (split_dir / "wav.scp").open("w", encoding="utf-8") as f:
        for utt_id, _speaker, path in entries:
            f.write(f"{utt_id} {path}\n")
    with (split_dir / "utt2spk").open("w", encoding="utf-8") as f:
        for utt_id, speaker, _path in entries:
            f.write(f"{utt_id} {speaker}\n")
    n_speakers = write_spk2utt(split_dir, entries)

    logger.info(
        "Wrote %s: %d utterances from %d speakers",
        split_dir.name,
        len(entries),
        n_speakers,
    )


def trial_path(data_root: Path, trial_name: str) -> Path:
    """Return the path of one converted trial list."""
    spec = _CFG["trials"][trial_name]
    return data_root / str(spec["split"]) / f"{trial_name}.trials"


class VoxCelebBuilder(DatasetBuilder):
    """Prepare Kaldi-style VoxCeleb manifests, trial lists, and augmentation.

    The corpus itself is not downloadable, so this builder only validates that
    the audio is in place and derives the manifests the recipe reads. Those
    manifests are Kaldi-style, the convention ESPnet uses throughout: sorted
    text files of whitespace-separated ``<key> <value...>`` records, namely
    ``wav.scp``, ``utt2spk``, and ``spk2utt``. :meth:`build` documents the
    resulting layout. The trial protocols are small text files and are fetched
    on demand.

    Splits are never concatenated on disk. Training on several of them is a
    matter of listing them under ``dataset.train``, which ESPnet3 merges through
    :class:`espnet3.components.data.dataset.CombinedDataset`. The one thing a
    merge does need is a shared speaker label space, so each union declared
    under ``builder.speaker_unions`` gets a directory holding only a ``spk2utt``.

    MUSAN and RIRS_NOISES are optional: when they are not found, the
    corresponding SCP files are skipped and training must disable augmentation.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the VoxCeleb audio and trial protocols are available."""
        recipe_root = Path(recipe_dir).resolve()
        try:
            resolve_source_root(recipe_root, source_dir=source_dir)
        except FileNotFoundError:
            return False
        download_root = recipe_root / _CFG["download_path"]
        return all(
            (download_root / str(spec["file_name"])).is_file()
            for spec in _CFG["trials"].values()
        )

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Validate the VoxCeleb audio and download the trial protocols.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional override pointing at the VoxCeleb root.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            FileNotFoundError: If the VoxCeleb audio trees are missing.
        """
        recipe_root = Path(recipe_dir).resolve()
        resolve_source_root(recipe_root, source_dir=source_dir)

        download_root = recipe_root / _CFG["download_path"]
        download_root.mkdir(parents=True, exist_ok=True)
        for trial_name, spec in _CFG["trials"].items():
            target = download_root / str(spec["file_name"])
            if target.is_file():
                continue
            logger.info("Downloading %s trial protocol to %s", trial_name, target)
            download_url(str(spec["url"]), target)

    def is_built(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return whether all manifests, label spaces, and trials are written."""
        data_root = Path(recipe_dir).resolve() / _CFG["data_path"]
        manifests_ready = all(
            (data_root / split / name).is_file()
            for split in _CFG["sources"]
            for name in MANIFESTS
        )
        # A label space is only ever a `spk2utt`; see `speaker_unions` in
        # `config.yaml` for why it holds nothing else.
        unions_ready = all(
            (data_root / union / "spk2utt").is_file()
            for union in _CFG["speaker_unions"]
        )
        trials_ready = all(
            trial_path(data_root, name).is_file() for name in _CFG["trials"]
        )
        return manifests_ready and unions_ready and trials_ready

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        n_workers: int | None = None,
        **_kwargs,
    ) -> None:
        """Convert the audio, then write the manifests, trials, and SCP files.

        VoxCeleb2 ships as AAC, which soundfile cannot read, so any source file
        that is not already WAV is decoded with ffmpeg before the manifests are
        written. The decoding runs through ``espnet3.parallel``, and the
        manifests are written only once every file they name exists.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional override pointing at the VoxCeleb root.
            n_workers: Overrides ``builder.parallel.n_workers`` for the
                conversion. Set it from a training config under
                ``create_dataset``.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            FileNotFoundError: If the VoxCeleb audio or a trial protocol is
                missing.
            RuntimeError: If a split contains no readable audio, or if ffmpeg is
                needed but missing or failing.

        Note:
            Everything is written under ``<recipe_dir>/data``, one directory
            per split plus the converted audio and the augmentation lists::

                data/
                  converted/            # WAVs decoded from the AAC sources
                    voxceleb2_dev/<speaker>/<video>/<utterance>.wav
                  voxceleb1_dev/        # VoxCeleb1 dev
                  voxceleb2_dev/        # VoxCeleb2 dev
                  voxceleb1_test/       # VoxCeleb1 test: the evaluation split
                    wav.scp             # <utt_id> <absolute path to the wav>
                    utt2spk             # <utt_id> <speaker_id>
                    spk2utt             # <speaker_id> <utt_id> <utt_id> ...
                    vox1_o.trials       # <label> <utt_id> <utt_id>, 1 == target
                  voxceleb12_dev/       # a label space, not a split
                    spk2utt             # speakers of VoxCeleb1 dev + VoxCeleb2 dev
                  musan_speech.scp      # one absolute MUSAN path per line
                  musan_noise.scp
                  musan_music.scp
                  rirs.scp              # one absolute RIRS_NOISES path per line

            Every real split holds the same three manifests; they are only
            expanded once above. ``wav.scp`` points an utterance ID at its
            audio and is what :class:`dataset.VoxCelebDataset` reads to load a
            waveform. ``utt2spk`` labels each utterance with its speaker, which
            the dataset attaches to every training sample. ``spk2utt`` is the
            inverse mapping; ``SpkPreprocessor`` reads it to turn those speaker
            strings into the integer class indices the AAMSoftmax head expects,
            so its line count is the ``spk_num`` the training config must
            declare. All of them are sorted by key.

            ``voxceleb12_dev`` is the odd one out: training on both development
            sets lists them both under ``dataset.train`` and lets
            ``CombinedDataset`` merge them, so no combined ``wav.scp`` is
            written. Only the union ``spk2utt`` is, because the label space has
            to span every split the AAMSoftmax head is trained on.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        data_root = recipe_root / _CFG["data_path"]
        convert_root = recipe_root / _CFG["convert_path"]

        entries_by_split = {}
        conversion_jobs: list[tuple[str, str]] = []
        for split, relative in _CFG["sources"].items():
            entries = scan_split(source_root / str(relative))
            entries, jobs = plan_conversions(convert_root, split, entries)
            entries_by_split[split] = entries
            conversion_jobs.extend(jobs)

        run_conversions(conversion_jobs, n_workers=n_workers)

        for split, entries in entries_by_split.items():
            write_manifests(data_root / split, entries)

        for union, parts in _CFG["speaker_unions"].items():
            merged = sorted(
                entry for part in parts for entry in entries_by_split[str(part)]
            )
            n_speakers = write_spk2utt(data_root / union, merged)
            logger.info(
                "Wrote %s/spk2utt: %d speakers over %s. Set `model.spk_num` to "
                "that number when training on these splits.",
                union,
                n_speakers,
                ", ".join(str(part) for part in parts),
            )

        for trial_name in _CFG["trials"]:
            self._build_trials(recipe_root, data_root, trial_name)

        self._build_augmentation_scp(data_root)

    def _build_trials(
        self, recipe_root: Path, data_root: Path, trial_name: str
    ) -> None:
        """Convert one official trial protocol into `<label> <utt1> <utt2>` lines."""
        spec = _CFG["trials"][trial_name]
        protocol = recipe_root / _CFG["download_path"] / str(spec["file_name"])
        if not protocol.is_file():
            raise FileNotFoundError(f"Trial protocol not found: {protocol}")

        known = set(
            line.split(maxsplit=1)[0]
            for line in (data_root / str(spec["split"]) / "wav.scp")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )

        output = trial_path(data_root, trial_name)
        n_trials = 0
        with output.open("w", encoding="utf-8") as f:
            for raw_line in protocol.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                label, utt1, utt2 = line.split()
                utt1 = utt1[: -len(AUDIO_SUFFIX)]
                utt2 = utt2[: -len(AUDIO_SUFFIX)]
                missing = [utt for utt in (utt1, utt2) if utt not in known]
                if missing:
                    raise RuntimeError(
                        f"Trial protocol {protocol} refers to utterances that are "
                        f"not in {spec['split']}: {', '.join(missing)}"
                    )
                f.write(f"{label} {utt1} {utt2}\n")
                n_trials += 1
        logger.info("Wrote %s: %d trials", output.name, n_trials)

    def _build_augmentation_scp(self, data_root: Path) -> None:
        """List the MUSAN and RIRS_NOISES files used for data augmentation."""
        musan_root = _env_path(_CFG["musan_env_var"])
        if musan_root is not None and musan_root.is_dir():
            for category in _CFG["musan_categories"]:
                paths = sorted((musan_root / str(category)).rglob("*.wav"))
                target = data_root / f"musan_{category}.scp"
                target.write_text(
                    "".join(f"{path.resolve()}\n" for path in paths),
                    encoding="utf-8",
                )
                logger.info("Wrote %s: %d files", target.name, len(paths))
        else:
            logger.warning(
                "%s is not set to a MUSAN directory, so musan_*.scp was not "
                "written. Set `noise_apply_prob: 0.0` in the training config to "
                "train without additive noise.",
                _CFG["musan_env_var"],
            )

        rir_root = _env_path(_CFG["rir_env_var"])
        if rir_root is not None and rir_root.is_dir():
            paths = sorted(
                path
                for room in _CFG["rir_rooms"]
                for path in (rir_root / str(room)).rglob("*.wav")
            )
            target = data_root / "rirs.scp"
            target.write_text(
                "".join(f"{path.resolve()}\n" for path in paths), encoding="utf-8"
            )
            logger.info("Wrote %s: %d files", target.name, len(paths))
        else:
            logger.warning(
                "%s is not set to a RIRS_NOISES directory, so rirs.scp was not "
                "written. Set `rir_apply_prob: 0.0` in the training config to "
                "train without reverberation.",
                _CFG["rir_env_var"],
            )
