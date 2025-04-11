"""Prepare ReazonSpeech for Japanese ASR"""

import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa

from utils import (
    SYMBOL_NA,
    SYMBOL_NOSPEECH,
    SYMBOLS_TIME,
    LongUtterance,
    Utterance,
    generate_long_utterances,
)


def find_duration(utt):
    wav_path = utt[0].wav_path.split()[4]
    if not Path(wav_path).is_file():
        logging.warning(f"missing audio file: {utt[0].utt_id}")
        return utt

    utt[0].end_time = librosa.get_duration(filename=wav_path)
    return utt


def find_durations(datas, nproc):
    pool = Pool(nproc)
    datas = pool.map(find_duration, datas)
    pool.close()
    return datas


def preprocess_text(text: str) -> str:
    # Note(jinchuan): not sure how should we treat a text
    # that is naturally with "<" or ">"
    if "<" in text or ">" in text:
        logging.warning(f"find an invalid text: {text}")
        text = text.replace("<", " ").replace(">", " ")
    return text


def collect_data(
    data_dir: Union[Path, str],
    split: str,
    prefix: str,
) -> List[List[Utterance]]:
    """Collect utterances in each long talk."""
    data_dir = Path(data_dir)
    lines = open(data_dir / f"{split}.tsv", "r").readlines()
    uttids = [line.strip().split()[0] for line in lines]
    trans = [" ".join(line.strip().split()[1:]) for line in lines]

    ans = []
    for idx, (uttid, tran) in enumerate(zip(uttids, trans)):
        path = data_dir / uttid

        uttid = uttid.replace(".flac", "").split("/")[1]

        text = preprocess_text(tran)
        if len(tran) == 0:
            logging.warning("empty string for {uttid}. Skip")
            continue

        ans.append(
            [
                Utterance(
                    utt_id=f"{prefix}_{uttid}",
                    wav_id=f"{prefix}_{uttid}",
                    wav_path=f"flac -c -d -s {str(path)} |",
                    start_time=0.0,
                    end_time=0.0,
                    lang="<jpn>",
                    task="<asr>",
                    text=text,
                    asr_text=text,
                )
            ]
        )

        if idx > 0 and idx % 100 == 0:
            logging.info(f"Done {idx} samples")
    return ans


def parse_args():
    parser = ArgumentParser(description="Prepare data.")
    parser.add_argument("--data_dir", type=Path, help="Path to raw data.")
    parser.add_argument(
        "--prefix", type=str, help="Prefix that will be added to utt id."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to save the output data.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["all"],
        help="Data splits to prepare.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=64,
        help="number of multi-processing to find the utterance duration",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        write_dir = args.output_dir / split
        write_dir.mkdir(parents=True, exist_ok=True)

        wavscp_fp = open(write_dir / "wav.scp", "w")  # wav-id wav-path
        segments_fp = open(
            write_dir / "segments", "w"
        )  # utt-id wav-id start-time end-time
        text_fp = open(write_dir / "text", "w")  # utt-id transcript
        textprev_fp = open(write_dir / "text.prev", "w")
        textctc_fp = open(
            write_dir / "text.ctc", "w"
        )  # text for ASR CTC w/o special tokens
        utt2spk_fp = open(write_dir / "utt2spk", "w")

        talks = collect_data(
            data_dir=args.data_dir,
            split=split,
            prefix=args.prefix,
        )

        talks = find_durations(talks, args.nproc)
        talks = [x for x in talks if x[0].end_time > 0.0]

        for talk in talks:
            for u in generate_long_utterances(talk):
                wavscp_fp.write(f"{u.wav_id} {u.wav_path}\n")
                segments_fp.write(
                    f"{u.utt_id} {u.wav_id} {u.start_time:.2f} {u.end_time:.2f}\n"
                )
                text_fp.write(f"{u.utt_id} {u.lang}{u.task}{u.text_with_time}\n")
                textprev_fp.write(f"{u.utt_id} {u.prev_text}\n")
                textctc_fp.write(f"{u.utt_id} {u.asr_text}\n")
                utt2spk_fp.write(f"{u.utt_id} {u.utt_id}\n")

        wavscp_fp.close()
        segments_fp.close()
        text_fp.close()
        textprev_fp.close()
        textctc_fp.close()
        utt2spk_fp.close()

    special_tokens = [
        SYMBOL_NA,
        SYMBOL_NOSPEECH,
        "<jpn>",
        "<asr>",
        *SYMBOLS_TIME,
    ]
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")
