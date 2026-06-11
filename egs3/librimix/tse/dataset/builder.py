"""LibriMix TSE dataset builder.

This module is responsible for:
1. Validating that the LibriSpeech source corpus is available (needed for
   mixture simulation).
2. Downloading WHAM! noise, LibriMix simulation scripts, and producing
   the simulated LibriMix split directories under ``<recipe_dir>/data/``.

The on-disk layout after a successful build looks like::

    <recipe_dir>/data/
    |-- LibriMix/            # simulation scripts and raw simulated audio
    |-- wham_noise/          # WHAM! noise corpus
    |-- 2mix_16k_max_dev_mix-clean/
    |   |-- wav.scp
    |   |-- spk1.scp
    |   |-- spk2.scp
    |   |-- text_spk1
    |   |-- text_spk2
    |   |-- utt2spk
    |   `-- spk2utt
    |-- 2mix_16k_max_test_mix-clean/
    |-- 2mix_16k_max_train_mix-clean/
    |-- 2mix_16k_max_train-100_mix-clean/
    `-- 2mix_16k_max_train-360_mix-clean/
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from importlib import resources
from pathlib import Path
from typing import Iterable, Optional

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url, extract_zip, setup_logger


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


# ---------------------------------------------------------------------------
# Path-resolution helpers
# ---------------------------------------------------------------------------
def resolve_librispeech_root(data_dir: str | Path) -> Path | None:
    """Resolve the LibriSpeech root directory.

    Traverses ``data_dir`` looking for a sub-directory that contains all
    expected LibriSpeech splits.

    Args:
        data_dir: Either the directory that contains ``LibriSpeech/`` or the
            ``LibriSpeech/`` directory itself.

    Returns:
        Path to the resolved ``LibriSpeech`` root, or ``None`` if not found.
    """
    p = Path(data_dir)
    all_dirs = [
        "train-clean-100",
        "train-clean-360",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ]
    for subdir in p.rglob(all_dirs[0]):
        root = subdir.parent
        if all((root / d).exists() for d in all_dirs):
            return root
    return None


def iter_source_candidates(
    recipe_root: str | Path,
    source_dir: str | Path | None,
    key: str = "source_env_var",
) -> Iterable[Path]:
    """Yield candidate directories that may contain LibriSpeech."""
    yield Path(recipe_root) / _CFG["dataset_path"]

    if source_dir is not None:
        yield Path(source_dir)

    env_var = str(_CFG[key])
    env_path = os.environ.get(env_var)
    if env_path:
        yield Path(env_path)


def resolve_source_root(
    recipe_root: Path,
    source_dir: str | Path | None = None,
) -> Path:
    """Resolve the usable LibriSpeech source root for this recipe."""
    checked: list[str] = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        checked.append(str(candidate))
        path = resolve_librispeech_root(candidate)
        if path is None:
            continue
        return path

    env_var = str(_CFG["source_env_var"])
    raise FileNotFoundError(
        "LibriSpeech source not found. Checked these locations:\n"
        + "\n".join(f"  - {path}" for path in checked)
        + "\n"
        + f"Place the corpus under <recipe_dir>/{_CFG['dataset_path']}/LibriSpeech "
        + f"or set {env_var} to the dataset root."
    )


def resolve_librimix_root(data_dir: str | Path, split: str) -> Path:
    """Resolve the LibriMix root directory.

    Args:
        data_dir: Either the directory that contains ``LibriMix/`` or the
            ``LibriMix/`` directory itself.
        split: The dataset split (e.g., ``2mix_16k_max_train_mix-both``) to check for
            existence of data files.

    Returns:
        Path to the resolved ``LibriMix`` root directory.

    Raises:
        FileNotFoundError: If the LibriMix root cannot be found.
    """
    p = Path(data_dir)
    if (p / "LibriMix").is_dir() and (p / split).is_dir():
        return p
    raise FileNotFoundError(
        "Could not find LibriMix root. Expected a directory containing both:\n"
        f"  - {p}/LibriMix/\n"
        f"  - {p}/{split}"
    )


def missing_required_splits(dataset_root: Path) -> list[str]:
    """Return required LibriMix split names that are missing from ``dataset_root``."""
    return [
        str(split)
        for split in _CFG["required_splits"]
        if not (dataset_root / str(split)).is_dir()
    ]


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------
def _download_and_extract_if_needed(
    *,
    url: str,
    dataset_dir: Path,
    extracted_dir: Path,
    archive_name: str,
    logger,
    step_percent: int = 5,
) -> None:
    """Download and extract one archive if not already present."""
    if extracted_dir and extracted_dir.exists():
        logger.info(f"Skip (already exists): {extracted_dir}")
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_dir / archive_name

    logger.info(f"Start processing: {archive_name}")

    if not archive_path.exists():
        download_url(
            url=url, dst_path=archive_path, logger=logger, step_percent=step_percent
        )
    elif not zipfile.is_zipfile(archive_path):
        logger.warning(
            f"Archive exists but is not a valid zip: {archive_path}. Re-downloading."
        )
        archive_path.unlink()
        download_url(
            url=url, dst_path=archive_path, logger=logger, step_percent=step_percent
        )
    else:
        logger.info(f"Archive exists, skipping download: {archive_path}")

    extract_zip(archive_path, dataset_dir, logger)
    logger.info(f"Finished: {archive_name}")


def _prepare_wham_noise(
    wham_noise: Optional[Path],
    dataset_dir: Path,
    logger,
    step_percent: int = 5,
) -> Path:
    """Ensure a WHAM! noise directory is available and return its path."""
    logger.info("Preparing WHAM! noise data...")
    if wham_noise is None:
        wham_noise = dataset_dir / "wham_noise"
        if wham_noise.exists():
            logger.info(f"Using existing noise data: {wham_noise}.")
            return wham_noise
    elif wham_noise.exists():
        if not os.access(wham_noise, os.W_OK):
            wham_noise_dst = dataset_dir / "wham_noise"
            wham_noise_dst.parent.mkdir(parents=True, exist_ok=True)
            if wham_noise_dst.exists():
                src_count = sum(1 for _ in wham_noise.rglob("*.wav"))
                dst_count = sum(1 for _ in wham_noise_dst.rglob("*.wav"))
                if src_count == dst_count:
                    logger.info(
                        f"wham_noise destination already exists: {wham_noise_dst}. "
                        "Skipping copy."
                    )
                    return wham_noise_dst
                raise RuntimeError(
                    f"wham_noise destination exists but file count differs: "
                    f"{wham_noise_dst}. Copy manually:\n"
                    f"    cp -r '{wham_noise}' '{wham_noise_dst}'"
                )
            logger.info(
                f"Copying wham_noise from '{wham_noise}' to '{wham_noise_dst}'..."
            )
            shutil.copytree(wham_noise, wham_noise_dst)
            return wham_noise_dst
        logger.info(f"Using provided wham_noise directory: {wham_noise}")
        return wham_noise

    dest_dir = (
        wham_noise if wham_noise.stem == "wham_noise" else dataset_dir / "wham_noise"
    )
    _download_and_extract_if_needed(
        url=(
            "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7."
            "s3.amazonaws.com/wham_noise.zip"
        ),
        dataset_dir=dataset_dir,
        extracted_dir=dest_dir,
        archive_name="wham_noise.zip",
        logger=logger,
        step_percent=step_percent,
    )
    return dest_dir


def _prepare_librimix_scripts(
    dataset_dir: Path, librimix_root: Path, logger, step_percent: int = 5
) -> None:
    """Download and extract the LibriMix simulation scripts."""
    logger.info("Preparing LibriMix simulation scripts...")
    _download_and_extract_if_needed(
        url="https://github.com/JorisCos/LibriMix/archive/refs/heads/master.zip",
        dataset_dir=dataset_dir,
        extracted_dir=librimix_root,
        archive_name="LibriMix.zip",
        logger=logger,
        step_percent=step_percent,
    )
    extracted_dir = dataset_dir / "LibriMix-master"
    if extracted_dir.exists() and not librimix_root.exists():
        extracted_dir.rename(librimix_root)


def _augment_wham_noise(wham_noise_dir: Path, librimix_root: Path, logger) -> None:
    """Run the LibriMix noise-augmentation script."""
    logger.info("Augmenting WHAM! noise for LibriMix simulation...")
    cwd = os.getcwd()
    os.chdir(librimix_root)
    command = [
        sys.executable,
        "scripts/augment_train_noise.py",
        "--wham_dir",
        str(wham_noise_dir),
    ]
    logger.info(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(
            f"Noise augmentation script failed with return code {result.returncode}"
        )
    os.chdir(cwd)


def _simulate_librimix(
    librimix_root: Path,
    wham_noise_dir: Path,
    librimix_outdir: Path,
    logger,
    num_spk: int = 2,
    mode: str = "min",
    fs: str = "8k",
) -> None:
    """Run the LibriMix mixture-simulation script."""
    logger.info(f"Simulating Libri{num_spk}Mix ({mode} mode, {fs} Hz)...")

    if librimix_outdir.exists():
        logger.info(f"Delting existing simulation output directory: {librimix_outdir}")
        shutil.rmtree(librimix_outdir)

    env_var = str(_CFG["source_env_var"])
    librispeech_dir = os.environ.get(env_var, "")
    if not librispeech_dir or resolve_librispeech_root(librispeech_dir) is None:
        raise EnvironmentError(
            f"Environment variable '{env_var}' is not set or invalid. "
            "Please point it to the LibriSpeech dataset root."
        )

    librispeech_dir = str(Path(librispeech_dir).absolute())
    wham_dir = str(wham_noise_dir.absolute())
    librimix_outdir = str(librimix_outdir.absolute())
    cwd = os.getcwd()
    os.chdir(librimix_root)
    args_to_pass = [
        # Librispeech as clean reference speech
        "--librispeech_dir",
        librispeech_dir,
        # WHAM! noise as noise sources
        "--wham_dir",
        wham_dir,
        # Simulation configs
        "--metadata_dir",
        f"metadata/Libri{num_spk}Mix",
        # Simulation output directory
        "--librimix_outdir",
        librimix_outdir,
        # Number of speakers per utterance in the simulated mixtures
        "--n_src",
        str(num_spk),
        # Sampling frequency for the simulated mixtures
        "--freqs",
        fs,
        # 'min' for trimming sources in the mixture to the same length
        # 'max' for keeping original lengths and zero-padding to the max length
        "--modes",
        mode,
        # Simulating multiple types of mixtures:
        #     mix_clean: mixing multi-speaker clean speech only
        #     mix_both: mixing multi-speaker clean speech with noise
        #     mix_single: mixing single-speaker speech with noise
        "--types",
        "mix_clean",
        "mix_both",
        "mix_single",
    ]
    command = [
        sys.executable,
        "scripts/create_librimix_from_metadata.py",
    ] + args_to_pass
    logger.info(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(
            f"LibriMix simulation script failed with return code {result.returncode}"
        )
    os.chdir(cwd)


def _prepare_librimix_transcripts(
    outdir: Path, uids: list[str], splits: list[str], num_spk: int
) -> None:
    """Write ``text_spkN`` files by reading LibriSpeech transcript files."""
    env_var = str(_CFG["source_env_var"])
    librispeech_dir = os.environ.get(env_var, "")
    if not librispeech_dir or resolve_librispeech_root(librispeech_dir) is None:
        raise EnvironmentError(
            f"Environment variable {env_var} is not set or invalid. "
            "Please point it to the LibriSpeech dataset root."
        )

    utt2txt: dict[str, str] = {}
    for split in splits:
        split_dir = Path(librispeech_dir) / split
        assert split_dir.exists(), split_dir
        for txt_path in split_dir.rglob("**/*.trans.txt"):
            with txt_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    s_uid, *words = line.split()
                    utt2txt[s_uid] = " ".join(words)

    for i in range(1, num_spk + 1):
        with (outdir / f"text_spk{i}").open("w", encoding="utf-8") as f:
            for uid in uids:
                s_uid = uid.split("_")[i - 1]
                f.write(f"{uid} {utt2txt[s_uid]}\n")


def _prepare_librimix_data(
    dataset_dir: Path,
    librimix_outdir: Path,
    logger,
    num_spk: int = 2,
    mode: str = "min",
    fs: str = "8k",
) -> None:
    """Write Kaldi-style scp/text files from LibriMix metadata CSVs."""
    logger.info(f"Preparing Libri{num_spk}Mix data files ({fs} Hz, {mode} mode)...")
    librimix_root = librimix_outdir / f"Libri{num_spk}Mix"
    data_dir = librimix_root / f"wav{fs}" / mode / "metadata"

    outdir: Path | None = None  # assigned each dset iteration; reused for sub-splits

    for typ in ("mix_clean", "mix_both", "mix_single"):
        for dset in ("dev", "test", "train"):
            if dset == "train":
                expected_files = list(data_dir.glob(f"mixture_train-*_{typ}.csv"))
                splits = ["train-clean-100", "train-clean-360"]
            else:
                expected_files = list(data_dir.glob(f"mixture_{dset}_{typ}.csv"))
                splits = [f"{dset}-clean", f"{dset}-other"]

            if not expected_files:
                raise FileNotFoundError(
                    f"Metadata files not found for split '{dset}' in {data_dir}. "
                    "Check that the simulation step completed successfully."
                )

            data: dict = {}
            for csv_file in expected_files:
                with csv_file.open("r") as f:
                    headers = f.readline().strip().split(",")
                    assert "mixture_ID" in headers, headers
                    for idx, line in enumerate(f, 1):
                        row = dict(zip(headers, line.strip().split(",")))
                        if len(row) < 5:
                            logger.warning(
                                f"Invalid line (#{idx}) in '{csv_file}': {line.strip()}"
                            )
                            continue
                        data[row["mixture_ID"]] = row

            uids = sorted(data.keys())
            assert len(uids) > 0, (typ, dset, expected_files)

            outdir = (
                dataset_dir / f"{num_spk}mix_{fs}_{mode}_{dset}_{typ.replace('_', '-')}"
            )
            outdir.mkdir(parents=True, exist_ok=True)

            with (outdir / "utt2spk").open("w") as f:
                for uid in uids:
                    f.write(f"{uid} {uid}\n")
            with (outdir / "spk2utt").open("w") as f:
                for uid in uids:
                    f.write(f"{uid} {uid}\n")
            with (outdir / "wav.scp").open("w") as f:
                for uid in uids:
                    f.write(f"{uid} {data[uid]['mixture_path']}\n")
            with (outdir / "spk1.scp").open("w") as f:
                for uid in uids:
                    f.write(f"{uid} {data[uid]['source_1_path']}\n")
            if typ != "mix_single":
                with (outdir / "spk2.scp").open("w") as f:
                    for uid in uids:
                        f.write(f"{uid} {data[uid]['source_2_path']}\n")
            if "noise_path" in data[uids[0]]:
                with (outdir / "noise1.scp").open("w") as f:
                    for uid in uids:
                        f.write(f"{uid} {data[uid]['noise_path']}\n")
            if num_spk == 3:
                with (outdir / "spk3.scp").open("w") as f:
                    for uid in uids:
                        f.write(f"{uid} {data[uid]['source_3_path']}\n")

            _prepare_librimix_transcripts(outdir, uids, splits, num_spk)
            logger.info(f"    Prepared {outdir} ({len(uids)} utterances).")

        if dset != "train" or outdir is None:
            continue

        # Extract train-100 and train-360 subsets from the full 'train' split
        for sset in ("train-100", "train-360"):
            csv_path = data_dir / f"mixture_{sset}_{typ}.csv"
            sub_uids: set[str] = set()
            with csv_path.open("r") as f:
                headers = f.readline().strip().split(",")
                assert "mixture_ID" in headers, headers
                for idx, line in enumerate(f, 1):
                    row = dict(zip(headers, line.strip().split(",")))
                    if len(row) < 5:
                        logger.warning(
                            f"Invalid line (#{idx}) in '{sset}' metadata: {line.strip()}"
                        )
                        continue
                    sub_uids.add(row["mixture_ID"])

            sub_outdir = (
                dataset_dir / f"{num_spk}mix_{fs}_{mode}_{sset}_{typ.replace('_', '-')}"
            )
            sub_outdir.mkdir(parents=True, exist_ok=True)

            for fname in (
                "utt2spk",
                "spk2utt",
                "wav.scp",
                "spk1.scp",
                "spk2.scp",
                "spk3.scp",
                "noise1.scp",
                "text_spk1",
                "text_spk2",
                "text_spk3",
            ):
                src = outdir / fname
                if not src.exists():
                    continue
                with (sub_outdir / fname).open("w") as fout:
                    with src.open("r") as fin:
                        for line in fin:
                            if line.split(maxsplit=1)[0] in sub_uids:
                                fout.write(line)


def _build_librimix(
    dataset_dir: Path,
    *,
    wham_noise: Optional[Path] = None,
    min_or_max: str = "max",
    num_spk: int = 2,
    sample_rate: str = "16k",
    step_percent: int = 5,
) -> None:
    """Full LibriMix build pipeline.

    Downloads WHAM! noise and LibriMix scripts, runs the mixture simulation,
    and writes Kaldi-style scp/text files under ``dataset_dir``.

    Args:
        dataset_dir: Destination directory (``<recipe_dir>/data/``).
        wham_noise: Optional pre-existing WHAM! noise directory.
        min_or_max: Mixture-length mode (``"max"`` or ``"min"``).
        num_spk: Number of speakers per mixture (2 or 3).
        sample_rate: Target sample rate (``"8k"`` or ``"16k"``).
        step_percent: Download progress logging granularity.
    """
    dataset_dir = Path(dataset_dir)
    logger = setup_logger(name="LibriMixTSEBuilder")
    librimix_root = dataset_dir / "LibriMix"

    wham_noise = _prepare_wham_noise(
        wham_noise=wham_noise,
        dataset_dir=dataset_dir,
        logger=logger,
        step_percent=step_percent,
    )
    _prepare_librimix_scripts(
        dataset_dir=dataset_dir,
        librimix_root=librimix_root,
        logger=logger,
        step_percent=step_percent,
    )
    _augment_wham_noise(
        wham_noise_dir=wham_noise, librimix_root=librimix_root, logger=logger
    )
    librimix_outdir = librimix_root.absolute() / "libri_mix"
    _simulate_librimix(
        librimix_root=librimix_root,
        wham_noise_dir=wham_noise,
        librimix_outdir=librimix_outdir,
        logger=logger,
        num_spk=num_spk,
        mode=min_or_max,
        fs=sample_rate,
    )
    _prepare_librimix_data(
        dataset_dir=dataset_dir,
        librimix_outdir=librimix_outdir,
        logger=logger,
        num_spk=num_spk,
        mode=min_or_max,
        fs=sample_rate,
    )
    logger.info("All splits processed successfully.")


# ---------------------------------------------------------------------------
# DatasetBuilder implementation
# ---------------------------------------------------------------------------


class LibriMixTSEBuilder(DatasetBuilder):
    """Download and simulate LibriMix mixtures for the TSE recipe.

    **Source** refers to the LibriSpeech corpus, which is required as clean
    reference speech during mixture simulation but is not downloaded
    automatically (it is too large).  Set the ``LIBRISPEECH`` environment
    variable (or pass ``librispeech_dir``) to point to the corpus root.

    **Build artifacts** are the simulated LibriMix split directories placed
    under ``<recipe_dir>/data/``.  WHAM! noise and the LibriMix simulation
    scripts are downloaded automatically during :meth:`build`.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the LibriSpeech corpus is available."""
        for candidate in iter_source_candidates(recipe_dir, source_dir):
            path = resolve_librispeech_root(candidate)
            if path is None:
                continue
            return path.exists()
        return False

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Validate that the LibriSpeech corpus is available.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional explicit path to the LibriSpeech corpus.
                Falls back to the ``LIBRISPEECH`` environment variable.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            FileNotFoundError: If the LibriSpeech corpus cannot be found.
        """
        env_var = str(_CFG["source_env_var"])
        candidate = source_dir or os.environ.get(env_var)
        if not candidate:
            raise FileNotFoundError(
                f"LibriSpeech source not found. Set the {env_var} environment "
                "variable to the directory containing the LibriSpeech corpus."
            )
        if resolve_librispeech_root(candidate) is None:
            raise FileNotFoundError(
                f"Could not find a valid LibriSpeech root under: {candidate}. "
                "Expected sub-directories: train-clean-100, train-clean-360, "
                "dev-clean, dev-other, test-clean, test-other."
            )

    def is_built(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        for candidate in iter_source_candidates(
            recipe_dir, source_dir, key="dataset_env_var"
        ):
            path = None
            for split in _CFG["required_splits"]:
                try:
                    path = resolve_librimix_root(candidate, str(split))
                except FileNotFoundError:
                    break
            else:
                if path is not None:
                    return path.exists()
        return False

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        wham_noise: Path | None = None,
        min_or_max: str = "max",
        num_spk: int = 2,
        sample_rate: str = "16k",
        **_kwargs,
    ) -> None:
        """Download WHAM! noise, LibriMix scripts, and simulate mixtures.

        Runs the full build pipeline: download WHAM! noise → download LibriMix
        simulation scripts → augment noise → run simulation → write scp files.

        Args:
            recipe_dir: Recipe root directory.  Simulated data lands in
                ``<recipe_dir>/data/``.
            librispeech_dir: Optional explicit path to the LibriSpeech corpus.
                Falls back to the ``LIBRISPEECH`` environment variable.
            wham_noise: Optional path to a pre-existing WHAM! noise directory.
                Downloaded automatically when omitted.
            min_or_max: Mixture length mode – ``"max"`` (zero-pad) or
                ``"min"`` (trim).  Defaults to ``"max"``.
            num_spk: Number of speakers per mixture (2 or 3).  Defaults to 2.
            sample_rate: Target sample rate (``"8k"`` or ``"16k"``).  Defaults
                to ``"16k"``.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            FileNotFoundError: If the LibriSpeech corpus is not found.
            EnvironmentError: If simulation scripts cannot be run.
        """
        env_var = str(_CFG["source_env_var"])
        candidate = resolve_source_root(
            recipe_dir, source_dir or os.environ.get(env_var)
        )
        os.environ[env_var] = str(candidate)
        _build_librimix(
            dataset_dir=Path(recipe_dir) / str(_CFG["dataset_path"]),
            wham_noise=wham_noise,
            min_or_max=min_or_max,
            num_spk=num_spk,
            sample_rate=sample_rate,
        )
