"""LibriMix downloader and simulator for the egs3 LibriMix recipe.

This module is used in two ways:
  1) As an ESPnet3 stage function via config (`create_dataset.func`), where the
     caller invokes :func:`create_dataset` directly with keyword arguments.
  2) As a standalone script (`python src/create_dataset.py ...`) for quick manual
     downloads during development.

Output directory structure (assuming 2-speaker, 16kHz, max mode simulation):

dataset_dir/
|-- wham_noise/  # (optional) processed WHAM noise directory
|-- LibriMix/    # Extracted LibriMix scripts and simulated data
|-- 2mix_16k_max_dev-mix-clean/
|   |-- spk1.scp
|   |-- spk2.scp
|   |-- noise1.scp
|   |-- text_spk1
|   |-- text_spk2
|   |-- utt2spk
|   `-- spk2utt
|-- 2mix_16k_max_dev-mix-both/
|-- 2mix_16k_max_dev-mix-single/
|-- 2mix_16k_max_test-mix-clean/
|-- 2mix_16k_max_test-mix-both/
|-- 2mix_16k_max_test-mix-single/
|-- 2mix_16k_max_train-mix-clean/
|-- 2mix_16k_max_train-mix-both/
`-- 2mix_16k_max_train-mix-single/
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

from espnet3.utils.download_utils import download_url, extract_zip, setup_logger


def _resolve_librispeech_root(data_dir: str | Path) -> Path:
    """Resolve the LibriSpeech root directory.

    Args:
        data_dir: Either the directory that contains ``LibriSpeech/`` or the
            ``LibriSpeech/`` directory itself.

    Returns:
        Path to the resolved ``LibriSpeech`` root directory.
        Or None if the LibriSpeech root cannot be found.
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


def download_and_extract_if_needed(
    *,
    split: str,
    url: str,
    dataset_dir: Path,
    extracted_dir: Path,
    archive_name: str,
    logger,
    step_percent: int = 5,
) -> None:
    """Download and extract one split if not already present."""
    if extracted_dir and extracted_dir.exists():
        logger.info(f"Skip split '{split}' (already exists): {extracted_dir}")
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_dir / archive_name

    logger.info(f"Start processing split: {split}")

    if not archive_path.exists():
        download_url(
            url=url,
            dst_path=archive_path,
            logger=logger,
            step_percent=step_percent,
        )
    elif not zipfile.is_zipfile(archive_path):
        # Check if zip file is valid before skipping extraction
        logger.warning(
            f"Archive already exists but is not a valid zip file: {archive_path}. "
            "Re-downloading the archive."
        )
        archive_path.unlink()  # Remove the invalid file
        download_url(
            url=url,
            dst_path=archive_path,
            logger=logger,
            step_percent=step_percent,
        )
    else:
        logger.info(f"Archive exists, skip download: {archive_path}")

    extract_zip(archive_path, dataset_dir, logger)
    logger.info(f"Finished split: {split}")


def prepare_wham_noise(
    wham_noise: Optional[Path],
    dataset_dir: Path,
    logger,
    step_percent: int = 5,
):
    logger.info("Preparing WHAM noise data...")
    if wham_noise is None:
        wham_noise = dataset_dir / "data/wham_noise"
        if wham_noise.exists():
            logger.info(f"Using existing noise data: {wham_noise}.")
            return wham_noise
    elif wham_noise.exists():
        if not os.access(wham_noise, os.W_OK):
            # copy to dataset_dir / "data/wham_noise"
            wham_noise_dst = dataset_dir / "data/wham_noise"
            wham_noise_dst.parent.mkdir(parents=True, exist_ok=True)
            if wham_noise_dst.exists():
                logger.info(
                    f"Wham_noise destination already exists: {wham_noise_dst}. "
                    "Skipping copy and using existing directory."
                )
            else:
                logger.info(
                    f"Copying wham_noise from '{wham_noise}' to '{wham_noise_dst}'..."
                )
                shutil.copytree(wham_noise, wham_noise_dst)
            return wham_noise_dst
        else:
            logger.info(f"Using provided wham_noise directory: {wham_noise}")
            return wham_noise

    dest_dir = wham_noise
    if wham_noise.stem != "wham_noise":
        dest_dir = dataset_dir / "wham_noise"
    download_and_extract_if_needed(
        split="",
        url="https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7."
        "s3.amazonaws.com/wham_noise.zip",
        dataset_dir=dataset_dir,
        extracted_dir=dest_dir,
        archive_name="wham_noise.zip",
        logger=logger,
        step_percent=step_percent,
    )
    return dest_dir


def prepare_librimix_scripts(
    dataset_dir: Path, librimix_root: Path, logger, step_percent: int = 5
):
    logger.info("Preparing LibriMix simulation scripts...")

    download_and_extract_if_needed(
        split="",
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


def augment_wham_noise(wham_noise_dir: Path, librimix_root: Path, logger):
    logger.info("Augmenting WHAM noise for LibriMix simulation...")

    cwd = os.getcwd()
    os.chdir(librimix_root)

    # Build the command list: [python_executable, script_name, arg1, arg2, ...]
    args_to_pass = ["--wham_dir", str(wham_noise_dir)]
    command = [sys.executable, "scripts/augment_train_noise.py"] + args_to_pass
    logger.info(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    logger.info("\nOutput from called_script.py:")
    logger.info(result.stdout)
    os.chdir(cwd)


def simulate_librimix(
    librimix_root: Path,
    wham_noise_dir: Path,
    librimix_outdir: Path,
    logger,
    num_spk=2,
    mode="min",
    fs="8k",
):
    logger.info(f"Simulating Libri{num_spk}Mix mixtures {mode} mode ({fs} Hz)")

    cwd = os.getcwd()
    os.chdir(librimix_root)

    # Build the command list: [python_executable, script_name, arg1, arg2, ...]
    librispeech_dir = os.environ.get("LIBRISPEECH", "")
    if not librispeech_dir or not _resolve_librispeech_root(librispeech_dir):
        raise EnvironmentError(
            "Environment variable LIBRISPEECH is not set. "
            "Please set it to the path of the LibriSpeech dataset for simulation."
        )
    args_to_pass = [
        # Librispeech as clean reference speech
        "--librispeech_dir",
        librispeech_dir,
        # WHAM! noise as noise sources
        "--wham_dir",
        str(wham_noise_dir),
        # Simulation configs
        "--metadata_dir",
        f"metadata/Libri{num_spk}Mix",
        # Simulation output directory
        "--librimix_outdir",
        str(librimix_outdir),
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
    command = [sys.executable, "scripts/augment_train_noise.py"] + args_to_pass
    logger.info(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    logger.info("\nOutput from called_script.py:")
    logger.info(result.stdout)
    os.chdir(cwd)


def prepare_librimix_data(
    dataset_dir: Path,
    librimix_outdir: Path,
    logger,
    num_spk=2,
    mode="min",
    fs="8k",
):
    logger.info(f"Preparing Libri{num_spk}Mix data files ({fs} Hz, {mode} mode)")
    librimix_root = librimix_outdir / f"Libri{num_spk}Mix"
    data_dir = librimix_root / f"wav{fs}" / mode / "metadata"

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
                    "Please check if the simulation step completed successfully."
                )

            data = {}
            for csv_file in expected_files:
                with csv_file.open("r") as f:
                    headers = f.readline().strip().split(",")
                    assert "mixture_ID" in headers, headers
                    for idx, line in enumerate(f, 1):
                        fields = line.strip().split(",")
                        dic = dict(zip(headers, fields))
                        if len(dic) < 5:
                            logger.warning(
                                f"Invalid line (#{idx}) in '{csv_file}': {line.strip()}"
                            )
                            continue
                        data[dic["mixture_ID"]] = dic

            uids = sorted(data.keys())
            assert len(uids) > 0, (typ, dset, expected_files, len(uids))

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
            prepare_librimix_transcripts(outdir, uids, splits, num_spk)
            logger.info(f"    Prepared {outdir} with {len(uids)} utterances.")

        if dset != "train":
            continue
        # Split train-100 and train-360 subsets from the 'train' split
        for sset in ("train-100", "train-360"):
            with (data_dir / f"mixture_{sset}_{typ}.csv").open("r") as f:
                headers = f.readline().strip().split(",")
                assert "mixture_ID" in headers, headers
                sub_uids = set()
                for idx, line in enumerate(f, 1):
                    fields = line.strip().split(",")
                    dic = dict(zip(headers, fields))
                    if len(dic) < 5:
                        logger.warning(
                            f"Invalid line (#{idx}) in '{sset}' metadata: {line.strip()}"
                        )
                        continue
                    sub_uids.add(dic["mixture_ID"])
            sub_uids = sorted((sub_uids))
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
                if not (outdir / fname).exists():
                    continue
                with (sub_outdir / fname).open("w") as f:
                    with (outdir / fname).open("r") as f_in:
                        for line in f_in:
                            uid = line.split(maxsplit=1)[0]
                            if uid in sub_uids:
                                f.write(line)


@lru_cache(maxsize=None)
def prepare_librimix_transcripts(
    outdir: Path, uids: list[str], splits: list[str], num_spk: int
):
    librispeech_dir = os.environ.get("LIBRISPEECH", "")
    if not librispeech_dir or not _resolve_librispeech_root(librispeech_dir):
        raise EnvironmentError(
            "Environment variable LIBRISPEECH is not set. Please set it to "
            "the path of the LibriSpeech dataset for reading transcripts."
        )

    utt2txt = {}
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


def create_dataset(
    dataset_dir: Path,
    *,
    wham_noise: Optional[Path] = None,
    min_or_max: str = "max",
    num_spk: int = 2,
    sample_rate: str = "16k",
    step_percent: int = 5,
) -> None:
    """Download (and extract) requested LibriMix splits into ``dataset_dir``.

    Args:
        dataset_dir: Destination directory. Archives are stored under this
            directory and extracted into ``dataset_dir / "LibriMix"``.
        wham_noise: Optional path to WHAM noise directory.
            If not provided, it will be downloaded and extracted.
            If provided and writable, it will be used directly.
            If provided but not writable, it will be copied and the copy will be used.
        min_or_max: Whether to use 'min' or 'max' mode for simulation.
        num_spk: Number of speakers for simulating speech mixtures (e.g., 2 or 3).
        sample_rate: Sample rate (in Hz) for simulated mixtures (e.g., '8k' or '16k').
        step_percent: Logging granularity for download progress.
    """
    dataset_dir = Path(dataset_dir)
    logger = setup_logger(name="create_dataset")

    librimix_root = dataset_dir / "LibriMix"

    wham_noise = prepare_wham_noise(
        wham_noise=wham_noise,
        dataset_dir=dataset_dir,
        logger=logger,
        step_percent=step_percent,
    )

    prepare_librimix_scripts(
        dataset_dir=dataset_dir,
        librimix_root=librimix_root,
        logger=logger,
        step_percent=step_percent,
    )

    augment_wham_noise(
        wham_noise_dir=wham_noise, librimix_root=librimix_root, logger=logger
    )

    librimix_outdir = librimix_root.absolute() / "libri_mix"
    simulate_librimix(
        librimix_root=librimix_root,
        wham_noise_dir=wham_noise,
        librimix_outdir=librimix_outdir,
        logger=logger,
        num_spk=num_spk,
        mode=min_or_max,
        fs=sample_rate,
    )

    prepare_librimix_data(
        dataset_dir=dataset_dir,
        librimix_outdir=librimix_outdir,
        logger=logger,
        num_spk=num_spk,
        mode=min_or_max,
        fs=sample_rate,
    )

    logger.info("All requested splits processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LibriMix to a directory.")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Target directory to store archives and extracted files.",
    )
    parser.add_argument(
        "--wham_noise",
        type=Path,
        default="",
        help="Path to the WHAM noise dataset (e.g., /path/to/WHAM). If not provided, "
        "it will be downloaded and extracted to dataset_dir/wham_noise.",
    )
    parser.add_argument(
        "--min_or_max",
        type=str,
        default="max",
        choices=["min", "max"],
        help="Whether to use min or max mode for simulation.",
    )
    parser.add_argument(
        "--num_spk",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of speakers for simulation (e.g., 2 or 3).",
    )
    parser.add_argument(
        "--sample_rate",
        type=str,
        default="16k",
        choices=["8k", "16k"],
        help="Sample rate for simulated mixtures (e.g., 8k or 16k).",
    )
    parser.add_argument(
        "--step_percent",
        type=int,
        default=5,
        help="Progress logging granularity (percent per log).",
    )

    args = parser.parse_args()
    create_dataset(
        args.dataset_dir,
        wham_noise=args.wham_noise,
        min_or_max=args.min_or_max,
        num_spk=args.num_spk,
        sample_rate=args.sample_rate,
        step_percent=args.step_percent,
    )
