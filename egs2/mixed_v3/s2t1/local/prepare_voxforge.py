"""Prepare CommonVoice data for multilingual ASR"""
import logging
from argparse import ArgumentParser
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
from iso639 import languages as iso_languages

from utils import (
    SYMBOL_NA,
    SYMBOL_NOSPEECH,
    SYMBOLS_TIME,
    LongUtterance,
    Utterance,
    generate_long_utterances,
)

NUM_PROC = 64


def preprocess_text(text: str) -> str:
    if "<" in text or ">" in text:
        logging.warning(f"Invalid text: {text}")
        text = text.replace("<", " ").replace(">", " ")
    text = " ".join(text.split())
    return text


def find_duration(utt):
    wav_path = utt[0].wav_path

    utt[0].end_time = librosa.get_duration(filename=wav_path.split()[2])
    utt[0].wav_path = wav_path

    return utt


def find_durations(datas):
    pool = Pool(NUM_PROC)
    datas = pool.map(find_duration, datas)
    pool.close()
    pool.join()
    return datas


def collect_data(
    data_dir: Union[Path, str], prefix: str, lang: str
) -> List[List[Utterance]]:
    text_dict = open(str(data_dir / "text")).readlines()
    text_dict = [line.strip().split() for line in text_dict]
    text_dict = {line[0]: " ".join(line[1:]) for line in text_dict}

    wav_dict = open(str(data_dir / "wav.scp")).readlines()
    wav_dict = [line.strip().split() for line in wav_dict]
    wav_dict = {line[0]: " ".join(line[1:]) for line in wav_dict}

    ans = []
    for uttid, text in text_dict.items():
        text = preprocess_text(text)
        if len(text) == 0:
            continue

        # Note(jinchuan): keep it in lower case
        text = text.lower()

        wav_path = wav_dict[uttid]

        ans.append(
            [
                Utterance(
                    utt_id=f"{prefix}_{uttid}",
                    wav_id=f"{prefix}_{uttid}",
                    wav_path=wav_path,
                    start_time=0.0,
                    end_time=0.0,
                    lang=f"<{lang}>",
                    task="<asr>",
                    text=text,
                    asr_text=text,
                )
            ]
        )

    ans = find_durations(ans)

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

    args = parser.parse_args()
    return args


def iso3_code(code2):
    lang = iso_languages.get(alpha2=code2)
    return lang.name, lang.part3


if __name__ == "__main__":
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_paths = (
        list(args.data_dir.glob("*test_*voxforge*"))
        + list(args.data_dir.glob("*dev_*voxforge*"))
        + list(args.data_dir.glob("*train_*voxforge*"))
    )
    print(all_paths, "all", flush=True)

    all_code3 = set()
    for path in all_paths:
        if path.is_file():
            continue

        if "_lid" in path.stem:  # without language-id
            continue

        print(path, flush=True)
        if (args.output_dir / path.stem / ".complete").is_file():
            logging.warning(f"{path.stem} processed before. Skip")
            continue

        language, code3 = iso3_code(path.stem.split("_")[1])
        all_code3.add(code3)
        logging.info(f"processing {path} | {language} | {code3}")

        write_dir = args.output_dir / path.stem
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
            data_dir=path,
            prefix=args.prefix,
            lang=code3,
        )
        logging.info(f"Done processing {path}. #utts: {len(talks)}")

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

        (args.output_dir / path.stem / ".complete").touch()

    special_tokens = [
        SYMBOL_NA,
        SYMBOL_NOSPEECH,
        *[f"<{x}>" for x in all_code3],
        "<asr>",
        *SYMBOLS_TIME,
    ]
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")
