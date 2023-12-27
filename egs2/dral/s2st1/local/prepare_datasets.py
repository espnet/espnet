"""download datasets and convert them into kaldi format"""
import logging
import multiprocessing
import os
import random
import re
import shutil
import tarfile
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import backoff
import pandas as pd
import requests
import typer
import yaml
from pydantic import BaseModel

_BACKOFF_NUM_RETRIES: int = 3
app = typer.Typer()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_CACHE_DIR: str = "cache"
_RECORDINGS: str = "recordings"
_FRAGMENTS_LONG: str = "fragments-long"
_FRAGMENTS_SHORT: str = "fragments-short"
_AUDIO_FOLDERS: List[str] = [_RECORDINGS, _FRAGMENTS_LONG, _FRAGMENTS_SHORT]
LOGGER = logging.getLogger(__name__)
_WAV_NAME_PATTERN_LONG: str = r"^[A-Z]{2}_\d{3}_#\d+\.wav$"
_WAV_NAME_PATTERN_SHORT: str = r"^[A-Z]{2}_\d{3}_\d+\.wav$"
_WAV_NAME_PATTERN_RECORDING: str = r"^[A-Z]{2}_\d{3}.wav$"
_SEED: int = 73
_WAV_SCP_FILE: str = "wav.scp"


def remove_suffix(s: str, suffix: str):
    if s.endswith(suffix):
        return s[: len(s) - len(suffix)]
    return s


class DatasetHelper:
    def __init__(self, disable_ssl_verification: bool = True):
        self._verify: bool = not disable_ssl_verification

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=_BACKOFF_NUM_RETRIES,
    )
    def download(self, url: str, destination: Path) -> Path:
        response = requests.get(url, stream=True, verify=self._verify)
        if response.status_code == 200:
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return destination
        else:
            raise requests.exceptions.RequestException(
                f"response.json() = {response.json()}; "
                f"response.status_code = {response.status_code}"
            )

    @staticmethod
    def uncompress(archive_path: Path, destination_path: Path) -> Path:
        destination_path.mkdir(exist_ok=True, parents=True)
        with tarfile.open(archive_path, "r") as tar_ref:
            tar_ref.extractall(destination_path)
        # the suffix name of release is inconsistent (.tgz or .tar.gz)
        if archive_path.suffix == ".tgz":
            subfolder = remove_suffix(archive_path.name, ".tgz")
        elif archive_path.suffix == ".tar":
            subfolder = remove_suffix(archive_path.name, ".tar")
        elif archive_path.name.endswith(".tar.gz"):
            subfolder = remove_suffix(archive_path.name, ".tar.gz")
        else:
            subfolder = archive_path.name
        return destination_path / subfolder


def download_thread_helper(url: str, dest_dir: Path) -> Tuple[str, Path]:
    dataset_helper = DatasetHelper()
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir: Path = Path(tmp_dir_str)
        tar_gz_path: Path = dataset_helper.download(url, tmp_dir / Path(url).name)
        return url, dataset_helper.uncompress(tar_gz_path, dest_dir)


class DatasetConfig(BaseModel):
    dataset_urls: List[str]


def _time_string_to_seconds(time_str):
    # Split the input string into days and time components
    days_str, time_components = time_str.split(" days ")

    # Parse the time components into hours, minutes, and seconds
    time_components = time_components.split(":")

    # Extract days, hours, minutes, and seconds
    days = int(days_str)
    hours = int(time_components[0])
    minutes = int(time_components[1])

    # Split seconds and microseconds
    tokens = time_components[2].split(".")
    if len(tokens) > 1:
        seconds_str, microseconds_str = tokens
    else:
        seconds_str = tokens[0]
        microseconds_str = "0"
    seconds = int(seconds_str)
    microseconds = int(microseconds_str)

    # Create a timedelta object
    duration = timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        microseconds=microseconds,
    )

    # Convert the duration to total seconds
    total_seconds = duration.total_seconds()

    return total_seconds


def _write_kaldi_files(
    raw_data_dir: Path,
    data_file_stem: str,
    df: pd.DataFrame,
    src_lang: str,
    tgt_lang: str,
    output_dir: Path,
):
    src_lang = src_lang.lower()
    tgt_lang = tgt_lang.lower()
    LOGGER.info(f"writing kaldi-style files: {data_file_stem} -> {output_dir}")
    assert data_file_stem in _AUDIO_FOLDERS, data_file_stem
    output_dir.mkdir(exist_ok=True, parents=True)
    # wav.scp
    df["audio_path"] = df["id_recordings"].apply(
        lambda x: str(raw_data_dir / data_file_stem / f"{x}.wav")
    )
    # add speaker id in id_recordings, trans_id, id_fragments
    df["id_recordings"] = df.apply(lambda x: f"{x.spk}-{x.id_recordings}", axis=1)
    df["trans_id"] = df.apply(lambda x: f"{x.spk}-{x.trans_id}", axis=1)
    df["id_fragments"] = df.apply(lambda x: f"{x.spk}-{x.id_fragments}", axis=1)
    src_wav_scp_file: str = f"wav.scp.{src_lang}"
    (
        df[["id_recordings", "audio_path"]]
        .drop_duplicates()
        .sort_values(["id_recordings"])
        .to_csv(output_dir / src_wav_scp_file, sep=" ", header=False, index=False)
    )
    os.symlink(f"wav.scp.{src_lang}", str(output_dir / _WAV_SCP_FILE))
    df["trans_audio_path"] = df["trans_id"].apply(
        lambda x: str(raw_data_dir / data_file_stem / f"{x}.wav")
    )
    (
        df[["trans_id", "trans_audio_path"]]
        .drop_duplicates()
        .sort_values(["trans_id"])
        .to_csv(output_dir / f"wav.scp.{tgt_lang}", sep=" ", header=False, index=False)
    )
    # utt2spk
    src_utt2spk = df[["id_recordings", "spk"]]
    tgt_utt2spk = df[["trans_id", "spk"]]
    tgt_utt2spk.columns = src_utt2spk.columns
    utt2spk = pd.concat([src_utt2spk, tgt_utt2spk], axis=0, ignore_index=True)
    utt2spk.drop_duplicates().sort_values(["id_recordings"]).to_csv(
        output_dir / "utt2spk", sep=" ", header=False, index=False
    )
    # spk2utt
    # Group by 'spk' and collect 'id' values into sets
    grouped_data = (
        utt2spk.groupby("spk")["id_recordings"]
        .apply(set)
        .reset_index()
        .sort_values(["spk"])
    )
    with (output_dir / "spk2utt").open("w") as spk2utt_out:
        for _, row in grouped_data.iterrows():
            spk = row["spk"]
            utt_str = " ".join(sorted(list(row["id_recordings"])))
            spk2utt_out.write(f"{spk} {utt_str}\n")
    # segments
    # if data_file_stem == _RECORDINGS:
    #     df["time_start"] = df["time_start"].apply(lambda s: _time_string_to_seconds(s))
    #     df["time_end"] = df["time_end"].apply(lambda s: _time_string_to_seconds(s))
    #     df[
    #         ["id_fragments", "id_recordings", "time_start", "time_end"]
    #     ].drop_duplicates().sort_values(["id_fragments", "id_recordings"]).to_csv(
    #         output_dir / "segments", sep=" ", header=False, index=False
    #     )
    # main table
    df.to_csv(
        output_dir / "main.tsv", sep="\t", header=False, index=False
    )
    LOGGER.info("completed!")


def _train_dev_split(df: pd.DataFrame):
    # deterministic random
    conv_ids = sorted(df["conv_id"].unique())
    random.seed(_SEED)
    random.shuffle(conv_ids)
    # split
    devtest_ratio = 0.1
    devtest_size = int(len(conv_ids) * devtest_ratio)
    devtest_ids = set(conv_ids[:devtest_size])
    dev_ids = set(conv_ids[devtest_size : devtest_size * 2])
    train_ids = set(conv_ids[devtest_size * 2 :])
    train_df = df[df["conv_id"].isin(train_ids)]
    dev_df = df[df["conv_id"].isin(dev_ids)]
    devtest_df = df[df["conv_id"].isin(devtest_ids)]
    return train_df, dev_df, devtest_df


@app.command(name="extract_fragments")
def extract_fragments(
    input_dir: Path,
    src_lang: str,
    tgt_lang: str,
    audio_type: str,
    output_dir: Path,
) -> Path:
    assert audio_type in {"fragments-long", "fragments-short"}
    file_stem: str
    if audio_type == "fragments-long":
        file_stem = _FRAGMENTS_LONG
    else:
        file_stem = _FRAGMENTS_SHORT
    conversation_df = pd.read_csv(
        input_dir / "conversation.csv",
        usecols=[
            "id",
            "participant_id_left_unique",
            "participant_id_right_unique",
            "trans_id",
        ],
    )
    fragment_df = pd.read_csv(
        input_dir / f"{file_stem}.csv",
        sep=",",
        header=0,
        usecols=[
            "id",
            "lang_code",
            "conv_id",
            "original_or_reenacted",
            "time_start",
            "time_end",
        ],
    )
    conversation_df["trans_lang_code"] = conversation_df["trans_id"].apply(
        lambda x: x.split("_", 1)[0]
    )
    df = pd.merge(
        conversation_df,
        fragment_df,
        left_on="id",
        right_on="conv_id",
        how="inner",
        suffixes=("_recordings", "_fragments"),
    )
    assert len(df) == len(
        fragment_df
    ), f"some audio rows do not exist in conversation.csv: {len(df)} vs {len(fragment_df)}"
    src_lang = src_lang.upper()
    tgt_lang = tgt_lang.upper()
    df = df[(df["lang_code"] == src_lang) & (df["trans_lang_code"] == tgt_lang)]
    LOGGER.info(f"{len(df)} segments for src_lang = {src_lang} & tgt_lang = {tgt_lang}")
    train_df, dev_df, devtest_df = _train_dev_split(df)
    _write_kaldi_files(
        input_dir,
        _RECORDINGS,
        train_df,
        src_lang,
        tgt_lang,
        output_dir / "train",
    )
    _write_kaldi_files(
        input_dir, _RECORDINGS, dev_df, src_lang, tgt_lang, output_dir / "dev"
    )
    _write_kaldi_files(
        input_dir,
        _RECORDINGS,
        devtest_df,
        src_lang,
        tgt_lang,
        output_dir / "test",
    )
    return output_dir


@app.command(name="extract_recordings")
def extract_recordings(
    input_dir: Path,
    src_lang: str,
    tgt_lang: str,
    time_type: str,
    output_dir: Path,
) -> Path:
    assert time_type in {"long", "short"}
    fragment_file_stem: str
    if time_type == "long":
        fragment_file_stem = _FRAGMENTS_LONG
    else:
        fragment_file_stem = _FRAGMENTS_SHORT
    # we join the conversation table with fragment table to get segment start and end time
    conversation_df = pd.read_csv(
        input_dir / "conversation.csv",
        usecols=[
            "id",
            "participant_id_left_unique",
            "participant_id_right_unique",
            "trans_id",
        ],
    )
    conversation_df["spk"] = conversation_df.apply(
        lambda x: "-".join(
            sorted(
                [
                    str(x["participant_id_left_unique"]),
                    str(x["participant_id_right_unique"]),
                ]
            )
        ),
        axis=1,
    )
    conversation_df = conversation_df.drop(
        ["participant_id_left_unique", "participant_id_right_unique"], axis=1
    )
    fragment_df = pd.read_csv(
        input_dir / f"{fragment_file_stem}.csv",
        sep=",",
        header=0,
        usecols=[
            "id",
            "lang_code",
            "conv_id",
            "original_or_reenacted",
            "time_start",
            "time_end",
        ],
    )
    conversation_df["trans_lang_code"] = conversation_df["trans_id"].apply(
        lambda x: x.split("_", 1)[0]
    )
    df = pd.merge(
        conversation_df,
        fragment_df,
        left_on="id",
        right_on="conv_id",
        how="inner",
        suffixes=("_recordings", "_fragments"),
    )
    assert len(df) == len(
        fragment_df
    ), f"some audio rows do not exist in conversation.csv: {len(df)} vs {len(fragment_df)}"
    src_lang = src_lang.upper()
    tgt_lang = tgt_lang.upper()
    df = df[(df["lang_code"] == src_lang) & (df["trans_lang_code"] == tgt_lang)]
    LOGGER.info(f"{len(df)} segments for src_lang = {src_lang} & tgt_lang = {tgt_lang}")
    train_df, dev_df, devtest_df = _train_dev_split(df)
    _write_kaldi_files(
        input_dir,
        _RECORDINGS,
        train_df,
        src_lang,
        tgt_lang,
        output_dir / "train",
    )
    _write_kaldi_files(
        input_dir, _RECORDINGS, dev_df, src_lang, tgt_lang, output_dir / "dev"
    )
    _write_kaldi_files(
        input_dir,
        _RECORDINGS,
        devtest_df,
        src_lang,
        tgt_lang,
        output_dir / "test",
    )
    return output_dir


@app.command(name="download")
def download(dataset_config: Path, output_dir: Path, no_cache: bool = False) -> Path:
    """
    download datasets and their fixes according to section 5 of
    https://www.cs.utep.edu/nigel/papers/dral-techreport2.pdf

    Args:
        dataset_config (Path): The path to the dataset configuration file.
        output_dir (Path): The path to store the merged dataset.
        no_cache (bool): whether to keep a cache

    Returns:
        Path: The path where the merged dataset is stored.
    """
    # read dataset config
    with dataset_config.open("r") as f_in:
        config: DatasetConfig = DatasetConfig.parse_obj(yaml.safe_load(f_in))
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        cache_dir: Path
        if no_cache:
            cache_dir = Path(tmp_dir_str)
        else:
            cache_dir = Path(_CACHE_DIR)
            cache_dir.mkdir(exist_ok=True, parents=True)
        # "download each installment, starting with DRAL-2.0, and unpack each into its own folder"
        num_cores = multiprocessing.cpu_count()
        LOGGER.info(f"downloading raw datasets: max_wokers = {num_cores}")
        ordered_dests: List[Tuple[str, Path]] = []
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            futures: List[Future[Path]] = []
            for url in config.dataset_urls:
                futures.append(executor.submit(download_thread_helper, url, cache_dir))
            results = wait(futures)
            # Wait for all threads to finish
            url2dest: Dict[str, Path] = {}
            for result in results.done:
                if result.exception() is not None:
                    raise result.exception()
                else:
                    url, destination_path = result.result()
                    url2dest[url] = destination_path
        for url in config.dataset_urls:
            ordered_dests.append(
                (
                    url,
                    url2dest[url],
                )
            )
        LOGGER.info("completed!")
        output_dir.mkdir(exist_ok=True, parents=True)
        LOGGER.info(f"populating metadatafiles from the last relese")
        LOGGER.info(f"ordered_dests = {ordered_dests}")
        # Discard all metadata (csv files) except those from the last release.
        for path in ordered_dests[-1][1].iterdir():
            if path.suffix == ".csv":
                shutil.move(str(path), str(output_dir / path.name))
        LOGGER.info("completed!")
        # Create directories for recordings/, fragments-long/, and fragments-short/
        recordings_dir = output_dir / _RECORDINGS
        fragments_long_dir = output_dir / _FRAGMENTS_LONG
        fragments_short_dir = output_dir / _FRAGMENTS_SHORT
        recordings_dir.mkdir(exist_ok=True, parents=True)
        fragments_long_dir.mkdir(exist_ok=True, parents=True)
        fragments_short_dir.mkdir(exist_ok=True, parents=True)
        # then populate each by copying over all files from the corresponding
        # directories in the releases.
        # Do this copying starting with 2.0 and working forward, to ensure that
        # any revised audio files overwrite the old ones.
        for url, path in ordered_dests:
            LOGGER.info(f"populating {url} from {path} to {output_dir}")
            for subfolder in _AUDIO_FOLDERS:
                src_dir = path / subfolder
                if src_dir.is_dir():
                    shutil.copytree(
                        str(src_dir),
                        str(output_dir / subfolder),
                        dirs_exist_ok=True,
                        copy_function=shutil.copy2,
                    )
            # release 6.1 does not follow the structure mentioned in their report
            # it scatters audios in the root directory
            for file in path.iterdir():
                if re.match(file.name, _WAV_NAME_PATTERN_LONG):
                    shutil.copy(str(file), fragments_long_dir / file.name)
                if re.match(file.name, _WAV_NAME_PATTERN_SHORT):
                    shutil.copy(str(file), fragments_short_dir / file.name)
                if re.match(file.name, _WAV_NAME_PATTERN_RECORDING):
                    shutil.copy(str(file), recordings_dir / file.name)
            LOGGER.info("completed!")
        LOGGER.info(f"audio data stored at {output_dir}")
    return output_dir


if __name__ == "__main__":
    app()
