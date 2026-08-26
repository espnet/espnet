"""VoxCeleb 1 & 2 dataset builder for speaker verification."""

from __future__ import annotations

import logging
import os
from importlib import resources
from pathlib import Path
from typing import Iterable

from espnet3.components.data.dataset_builder import DatasetBuilder
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


def scan_split(split_dir: Path) -> list[tuple[str, str, Path]]:
    """Index one VoxCeleb audio tree as ``(utt_id, speaker, path)`` entries.

    VoxCeleb is laid out as ``<speaker>/<video>/<utterance>.wav`` and the
    utterance ID is that relative path without its suffix, which is also how
    the official trial lists refer to utterances.

    Args:
        split_dir: Directory holding one VoxCeleb split.

    Returns:
        Entries sorted by utterance ID.

    Raises:
        RuntimeError: If the directory contains no WAV files.
    """
    entries = []
    for path in split_dir.rglob(f"*{AUDIO_SUFFIX}"):
        speaker, video, utterance = path.parts[-3:]
        utt_id = f"{speaker}/{video}/{utterance[: -len(AUDIO_SUFFIX)]}"
        entries.append((utt_id, speaker, path.resolve()))

    if not entries:
        raise RuntimeError(
            f"No {AUDIO_SUFFIX} files found under: {split_dir}. VoxCeleb2 ships "
            "as AAC, so convert it to 16 kHz WAV before running this recipe."
        )
    return sorted(entries)


def write_manifests(split_dir: Path, entries: list[tuple[str, str, Path]]) -> None:
    """Write ``wav.scp``, ``utt2spk``, and ``spk2utt`` for one split."""
    split_dir.mkdir(parents=True, exist_ok=True)
    spk2utt: dict[str, list[str]] = {}
    for utt_id, speaker, _path in entries:
        spk2utt.setdefault(speaker, []).append(utt_id)

    with (split_dir / "wav.scp").open("w", encoding="utf-8") as f:
        for utt_id, _speaker, path in entries:
            f.write(f"{utt_id} {path}\n")
    with (split_dir / "utt2spk").open("w", encoding="utf-8") as f:
        for utt_id, speaker, _path in entries:
            f.write(f"{utt_id} {speaker}\n")
    with (split_dir / "spk2utt").open("w", encoding="utf-8") as f:
        for speaker in sorted(spk2utt):
            f.write(f"{speaker} {' '.join(spk2utt[speaker])}\n")

    logger.info(
        "Wrote %s: %d utterances from %d speakers",
        split_dir.name,
        len(entries),
        len(spk2utt),
    )


def trial_path(data_root: Path, trial_name: str) -> Path:
    """Return the path of one converted trial list."""
    spec = _CFG["trials"][trial_name]
    return data_root / str(spec["split"]) / f"{trial_name}.trials"


class VoxCelebBuilder(DatasetBuilder):
    """Prepare VoxCeleb manifests, trial lists, and augmentation sources.

    The corpus itself is not downloadable, so this builder only validates that
    the audio is in place and derives the Kaldi-style manifests the recipe
    reads. The trial protocols are small text files and are fetched on demand.

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
        """Return whether all manifests and trial lists have been written."""
        data_root = Path(recipe_dir).resolve() / _CFG["data_path"]
        splits = list(_CFG["sources"]) + list(_CFG["combined"])
        manifests_ready = all(
            (data_root / split / name).is_file()
            for split in splits
            for name in MANIFESTS
        )
        trials_ready = all(
            trial_path(data_root, name).is_file() for name in _CFG["trials"]
        )
        return manifests_ready and trials_ready

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Write the manifests, trial lists, and augmentation SCP files.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional override pointing at the VoxCeleb root.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            FileNotFoundError: If the VoxCeleb audio or a trial protocol is
                missing.
            RuntimeError: If a split contains no WAV files.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        data_root = recipe_root / _CFG["data_path"]

        entries_by_split = {}
        for split, relative in _CFG["sources"].items():
            entries = scan_split(source_root / str(relative))
            entries_by_split[split] = entries
            write_manifests(data_root / split, entries)

        for split, parts in _CFG["combined"].items():
            combined = sorted(
                entry for part in parts for entry in entries_by_split[str(part)]
            )
            write_manifests(data_root / split, combined)

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
