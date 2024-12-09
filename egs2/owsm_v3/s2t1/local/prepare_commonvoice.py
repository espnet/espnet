"""Prepare CommonVoice data for multilingual ASR"""

import csv
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

# To read large files
csv.field_size_limit(50000 * 1024 * 1024)


def preprocess_text(text: str) -> str:
    if "<" in text or ">" in text:
        logging.warning(f"Invalid text: {text}")
        text = text.replace("<", " ").replace(">", " ")
    text = " ".join(text.split())
    return text


def transform_into_talks(datas, data_dir, prefix, lang):
    ans = []
    for d in datas:
        text = preprocess_text(d["sentence"])
        if text is None or len(text) > 2000:
            continue

        path = data_dir / "clips" / d["path"]
        if not path.is_file():
            continue

        # We don't know the length of each utterances.
        # Will fill it when all other info is collected.
        ans.append(
            [
                Utterance(
                    utt_id=f"{prefix}_{d['client_id']}",
                    wav_id=f"{prefix}_{d['path']}",
                    wav_path=f"ffmpeg -i {str(path)} -f wav -ar 16000 -ab 16 -ac 1 - |",
                    start_time=0.0,
                    end_time=0.0,
                    lang=f"<{lang}>",
                    task="<asr>",
                    text=text,
                    asr_text=text,
                )
            ]
        )
    return ans


def find_duration(utt):
    if Path(utt[0].wav_path).is_file():
        logging.warning(f"cannot find the path {utt[0].wav_path}")
        return None

    utt[0].end_time = librosa.get_duration(filename=utt[0].wav_path.split()[2])
    return utt


def find_durations(datas, nproc):
    pool = Pool(nproc)
    datas = pool.map(find_duration, datas)
    pool.close()
    datas = [d for d in datas if d is not None]
    return datas


def collect_data(
    data_dir: Union[Path, str],
    prefix: str,
    lang: str,
    nproc: int,
) -> List[List[Utterance]]:
    validated_data = [
        x for x in csv.DictReader(open(str(data_dir / "validated.tsv")), delimiter="\t")
    ]
    dev_data = [
        x for x in csv.DictReader(open(str(data_dir / "dev.tsv")), delimiter="\t")
    ]
    test_data = [
        x for x in csv.DictReader(open(str(data_dir / "test.tsv")), delimiter="\t")
    ]

    dev_ids = {x["client_id"]: None for x in dev_data}
    test_ids = {x["client_id"]: None for x in test_data}
    train_data = [
        x
        for x in validated_data
        if x["client_id"] not in dev_ids and x["client_id"] not in dev_ids
    ]

    logging.info("start to transform into talks")
    train_data = transform_into_talks(train_data, data_dir, prefix, lang)
    dev_data = transform_into_talks(dev_data, data_dir, prefix, lang)
    test_data = transform_into_talks(test_data, data_dir, prefix, lang)

    logging.info("Start finding durations")
    train_data = find_durations(train_data, nproc)
    dev_data = find_durations(
        dev_data,
        nproc,
    )
    test_data = find_durations(test_data, nproc)

    return train_data, dev_data, test_data


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
        "--nproc",
        type=int,
        default=64,
        help="number of multi-processing to find the utterance duration",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

    # map CommonVoice language-id to ISO-693-3 standard code.
    language_map = open("local/cv-iso-693-3.txt").readlines()
    language_map = [x.strip().split() for x in language_map]
    language_map = {x[0]: x[1] for x in language_map}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for lang in args.data_dir.iterdir():
        if lang.is_file():
            continue

        lang = lang.stem
        lang_iso = language_map[lang]

        logging.info(
            f"For language {lang} (CommonVoice) , a.k.a., {lang_iso} (whisper)"
        )

        if (args.output_dir / lang_iso / ".complete").is_file():
            logging.info(f"language {lang} has been processed. Skip.")
            continue

        train, dev, test = collect_data(
            data_dir=args.data_dir / lang,
            prefix=args.prefix,
            lang=lang_iso,
            nproc=args.nproc,
        )

        logging.info(f"#Train: {len(train)} | #Dev: {len(dev)} | #Test: {len(test)}")

        for split, talks in zip(["train", "dev", "test"], [train, dev, test]):
            # (Jinchuan): some languages in CommonVoice share the same language-id
            # so we have to use lang_iso + lang to distinguish them
            write_dir = args.output_dir / (lang_iso + "-" + lang) / split
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

        (args.output_dir / (lang_iso + "-" + lang) / ".complete").touch()

    special_tokens = [
        SYMBOL_NA,
        SYMBOL_NOSPEECH,
        *[f"<{x}>" for x in language_map.values()],
        "<asr>",
        *SYMBOLS_TIME,
    ]
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")
