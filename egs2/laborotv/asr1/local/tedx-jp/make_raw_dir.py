# !usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make a kaldi-format data directory from pairs of .wav and .vtt files
in a input directory
"""

import sys
import re
import glob
import argparse

import shutil
from pathlib import Path

import pandas as pd

import jaconv

import subprocess

from logging import basicConfig, getLogger

logger = getLogger(__name__)


def convert_timestamp_to_seconds(timestamp):
    """Convert timestamp (hh:mm:ss) to seconds

    Args:
        timestamp (str): timestamp text in `hh:mm:ss` format

    Returns:
        float: timestamp converted to seconds
    """
    timestamp_split = timestamp.split(":")
    hour = int(timestamp_split[0])
    minute = int(timestamp_split[1])
    second = float(timestamp_split[2])

    second_sum = hour * 3600 + minute * 60 + second
    return second_sum


def clean_text(text):
    """Clean up one line of text in subtitles

    Args:
        text (str): input text

    Returns:
        str: output text
    """
    # Remove \U+200B
    text = text.replace("\u200b", "")

    # Han-kaku => Zen-kaku
    text = jaconv.h2z(text, kana=False, ascii=True, digit=True)

    # Remove first line
    text = re.sub(r"翻訳：.*", "", text)
    text = re.sub(r"字幕：.*", "", text)

    # Remove words in foreign languages
    text = re.sub(r"（.*語）.*", "", text)

    # Remove words added only in subtitles
    text = re.sub(r"（.*?）", "", text)  # （笑） etc.
    text = re.sub(r"［.*?］", "", text)  # [の], [を] etc.

    # Remove brackets
    text = re.sub(r"「|」|『|』|［|］|\"|“|”|＂", " ", text)

    # Remove/fix punctuation marks & Special symbols
    text = re.sub(r"！|？|、|。|…|—|・・・|．．．", " ", text)
    text = re.sub(r"〜", "ー", text)
    text = re.sub(r"♪", "", text)  # Music
    text = re.sub(r"―$", "", text)  # Music

    # Fix trailing white spaces
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+$|^\s+", "", text)

    return text


def load_vtt_as_dataframe(vtt_path):
    """Load a vtt-format subtitle file to a DataFrame

    Args:
        vtt_path (str): path to a .vtt file

    Returns:
        pandas.DataFrame: table containing subtitle info. with columns:
            - text_original (str): original subtitle text
            - text (str): text after simple cleanups
            - start_sec (float)
            - end_sec (float)
    """
    records = []
    with open(vtt_path, "r") as f:
        read_line = False
        current_text = []
        for line in f:
            line = line.rstrip()

            # Time
            if "-->" in line:
                start_time, _, end_time = line.split()[:3]
                read_line = True

            # End of segments
            elif re.match(r"^\s*$", line):
                if current_text:
                    entry = [start_time, end_time, " ".join(current_text)]
                    records.append(entry)
                    current_text = []
                read_line = False

            # Text
            elif read_line:
                current_text.append(line)
    df = pd.DataFrame.from_records(
        records, columns=["start_time", "end_time", "text_original"]
    )
    df["text"] = df["text_original"].apply(clean_text)
    df["start_sec"] = df["start_time"].apply(convert_timestamp_to_seconds)
    df["end_sec"] = df["end_time"].apply(convert_timestamp_to_seconds)
    return df


def load_video_ids(csv_path):
    """Load a list of video-ids from a csv file with a column named 'id'

    Args:
        csv_path (str): csv file with a column named 'id'

    Returns:
        list: list of video ids
    """
    df = pd.read_csv(csv_path)
    return list(df["id"])


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src_data_dir", type=Path, help="Path to directory containing .wav & .vtt files"
    )
    parser.add_argument("dst_dir", type=Path, help="")
    parser.add_argument(
        "--copy-wav",
        action="store_true",
    )
    parser.add_argument("--video-csv")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        dest="log_level",
        const="DEBUG",
        default="INFO",
    )
    args = parser.parse_args()

    basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        level=args.log_level,
    )

    args.dst_dir.mkdir(exist_ok=True, parents=True)

    if args.copy_wav:
        copied_wav_dir = args.dst_dir / "wav"
        copied_wav_dir.mkdir(exist_ok=True, parents=True)

    # Files to write
    wav_scp = (args.dst_dir / "wav.scp").open("w")
    segments = (args.dst_dir / "segments").open("w")
    text = (args.dst_dir / "text").open("w")
    utt2spk = (args.dst_dir / "utt2spk").open("w")

    # Collect video_ids to use
    if args.video_csv:
        target_video_ids = load_video_ids(args.video_csv)
    else:
        target_video_ids = None

    for subtitle_path in sorted(glob.glob(str(args.src_data_dir / "*.ja.vtt"))):
        video_id = Path(subtitle_path).name.split(".")[0]

        # Skip if necessary
        if target_video_ids and video_id not in target_video_ids:
            logger.warning(
                f"Skipped {video_id} as it is not listed in {args.video_csv}."
            )
            continue

        wav_path = Path(subtitle_path).parent / f"{video_id}.wav"
        if not wav_path.exists():
            logger.error(f"{wav_path} does not exist.")
            sys.exit(1)

        if args.copy_wav:
            copied_wav_path = copied_wav_dir / f"{video_id}.wav"
            shutil.copy(wav_path, copied_wav_path)
            wav_path = copied_wav_path

        spk = f"{video_id}_spk"

        print(f"{video_id} sox {wav_path} -c 1 -r 16000 -t wav - |", file=wav_scp)

        df = load_vtt_as_dataframe(subtitle_path)
        df = df.query("text != ''")  # Remove empty lines

        for idx, row in df.iterrows():
            # if row.text == "":
            #     continue
            utt_id = f"{video_id}_{idx:04d}"
            print(" ".join([utt_id, spk]), file=utt2spk)
            print(" ".join([utt_id, row.text]), file=text)
            print(
                " ".join(
                    [utt_id, video_id, f"{row.start_sec:.2f}", f"{row.end_sec:.2f}"]
                ),
                file=segments,
            )

    spk2utt = (args.dst_dir / "spk2utt").open("w")
    subprocess.run(
        ["utils/utt2spk_to_spk2utt.pl", str((args.dst_dir / "utt2spk"))],
        stdout=spk2utt,
    )

    logger.info(f"done making {args.dst_dir}")


if __name__ == "__main__":
    main()
