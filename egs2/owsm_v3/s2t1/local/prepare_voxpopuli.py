"""Prepare VoxPopuil data for multilingual ASR & ST."""

import csv
import gzip
import json
import logging
import urllib
from argparse import ArgumentParser
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from iso639 import languages as iso_languages
from torchaudio.datasets.utils import download_url

from utils import (
    SYMBOL_NA,
    SYMBOL_NOSPEECH,
    SYMBOLS_TIME,
    LongUtterance,
    Utterance,
    generate_long_utterances,
)

# Note(jinchuan): This increases the upper limit of csv
# some large label files cannot be loaded without this
csv.field_size_limit(50000 * 1024 * 1024)

DOWNLOAD_BASE_URL = "https://dl.fbaipublicfiles.com/voxpopuli"
asr_langs = [
    "en",
    "de",
    "fr",
    "es",
    "pl",
    "it",
    "ro",
    "hu",
    "cs",
    "nl",
    "fi",
    "hr",
    "sk",
    "sl",
    "et",
    "lt",
]
st_src_langs = asr_langs
# Note(jinchuan): only 3 languages have tgt_text.
# Can only use them for supervised training
st_tgt_langs_human_transcription = ["en", "fr", "es"]


def toiso(lid):
    return iso_languages.get(alpha2=lid).part3


def parse_src_id(id_):
    event_id, utt_id = id_.split("_", 1)
    event_id, lang = event_id.rsplit("-", 1)
    return event_id, lang, utt_id


def preprocess_text(text: str) -> str:
    # Note(jinchuan): not sure how should we treat a text
    # that is naturally with "<" or ">"
    if "<" in text or ">" in text:
        logging.warning(f"find an invalid text: {text}")
        text = text.replace("<", " ").replace(">", " ")
    text = " ".join(text.split())
    return text


def collect_data_asr(
    data_dir: Union[Path, str],
    split: str,
    prefix: str,
    src: str,
) -> List[List[Utterance]]:
    in_root = data_dir / "raw_audios" / "original"
    csv_root = data_dir / "transcribed_data" / src
    csv_root.mkdir(exist_ok=True, parents=True)

    # Get metadata TSV
    url = f"{DOWNLOAD_BASE_URL}/annotations/asr/asr_{src}.tsv.gz"
    tsv_path = csv_root / Path(url).name
    if not tsv_path.exists():
        try:
            download_url(url, csv_root.as_posix(), Path(url).name)
        except urllib.error.HTTPError:
            logging.warning(f"cannot download {url}. Skip")
            return []

    with gzip.open(tsv_path, "rt") as f:
        metadata = [x for x in csv.DictReader(f, delimiter="|")]
        metadata = [x for x in metadata if x["split"] == split]

    # Get talks
    talks = {}
    for r in metadata:
        event_id = r["session_id"]
        if event_id not in talks:
            talks[event_id] = []

        year = event_id[:4]
        audio_path = in_root / year / f"{event_id}_original.ogg"
        assert audio_path.is_file()

        # Note(jinchuan): Try to reserve the original text.
        # adopt original_text if provided, otherwise normed_text.
        if len(r["original_text"].strip()) > 0:
            text = r["original_text"]
        elif len(r["normed_text"].strip()) > 0:
            text = r["normed_text"]
        else:
            logging.warning(f"text is missing: {r}")
            continue
        text = preprocess_text(text)

        # Note(jinchuan): it has two time-stamps: "vad" and
        # "start_time" & "end_time". We adopt "vad" follwoing
        # the original VoxPopuli script.
        start_time = literal_eval(r["vad"])[0][0]
        end_time = literal_eval(r["vad"])[-1][-1]
        if end_time - start_time < 0.1:
            logging.warning(f"the utterance is too short. Skip: {r}")
            continue

        iso_src = toiso(src)
        path_template = "ffmpeg -i {} -ac 1 -ar 16000 -f wav - |"
        talks[event_id].append(
            Utterance(
                utt_id=f"{prefix}_asr_{event_id}_{r['id_']}",
                wav_id=f"{prefix}_asr_{event_id}",
                wav_path=path_template.format(str(audio_path.resolve())),
                start_time=start_time,
                end_time=end_time,
                lang=f"<{iso_src}>",
                task="<asr>",
                text=text,
                asr_text=text,
            )
        )

    return list(talks.values())


def collect_data_st(
    data_dir: Union[Path, str],
    split: str,
    prefix: str,
    src: str,
    tgt: str,
) -> List[List[Utterance]]:
    in_root = data_dir / "raw_audios" / tgt
    asr_root = data_dir / "transcribed_data" / src
    st_root = data_dir / "transcribed_data" / tgt
    asr_root.mkdir(exist_ok=True, parents=True)
    st_root.mkdir(exist_ok=True, parents=True)

    # Get metadata TSV
    url = f"{DOWNLOAD_BASE_URL}/annotations/asr/asr_{src}.tsv.gz"
    tsv_path = asr_root / Path(url).name
    if not tsv_path.exists():
        try:
            download_url(url, asr_root.as_posix(), Path(url).name)
        except urllib.error.HTTPError:
            logging.warning(f"cannot download {url}. Skip")
            return []

    with gzip.open(tsv_path, "rt") as f:
        src_metadata = [x for x in csv.DictReader(f, delimiter="|")]
        src_metadata = {
            "{}-{}".format(r["session_id"], r["id_"]): (
                (
                    r["original_text"]
                    if len(r["original_text"].strip()) > 0
                    else r["normed_text"]
                ),
                r["split"],
            )
            for r in src_metadata
        }

    url = f"{DOWNLOAD_BASE_URL}/annotations/s2s/s2s_{tgt}_ref.tsv.gz"
    # Note(jinchuan): st label is irrelevent to "src" lang.
    # Don't download it for multiple times
    tsv_path = st_root / Path(url).name
    if not tsv_path.exists():
        try:
            download_url(url, st_root.as_posix(), Path(url).name)
        except urllib.error.HTTPError:
            logging(f"cannot download {url}. Skip")
            return []
    with gzip.open(tsv_path, "rt") as f:
        tgt_metadata = [x for x in csv.DictReader(f, delimiter="\t")]

    talks = {}
    for r in tgt_metadata:
        src_id = r["id"]
        event_id, _src, utt_id = parse_src_id(src_id)
        # Note(jinchuan): not sure if the segments are contiguous.
        if _src != src:
            continue

        # Note(Jinchuan): ST data do not have explicit split. Inherit it from ASR data
        src_text, _split = src_metadata.get(src_id, (None, None))
        if src_text is None:
            logging.warning(f"src: {src} | tgt: {tgt} | missing key in src: {src_id}")
            continue

        if _split != split:
            continue

        src_text = preprocess_text(src_text)
        if len(src_text.strip()) == 0:
            logging.warning(f"example {src_id} has empty src text. Skip. {r}")
            continue

        year = event_id[:4]
        audio_path = in_root / year / f"{event_id}_{tgt}.ogg"
        assert audio_path.is_file()

        tgt_text = r["tgt_text"]
        tgt_text = preprocess_text(tgt_text)
        if len(tgt_text.strip()) == 0:
            logging.warning(f"example {src_id} has empty tgt text. Skip. {r}")
            continue

        if event_id not in talks:
            talks[event_id] = []

        if float(r["end_time"]) - float(r["start_time"]) < 0.1:
            logging.warning(f"the utterance is too short. Skip: {r}")

        # Note(jinchuan): Not sure if "event_id" would overlap across
        # languages. So add src2tgt tag to wav_id to exclude this risk
        iso_src, iso_tgt = toiso(src), toiso(tgt)
        path_template = "ffmpeg -i {} -ac 1 -ar 16000 -f wav - |"
        talks[event_id].append(
            Utterance(
                utt_id=f"{prefix}_st_{iso_src}2{iso_tgt}_{event_id}_{utt_id}",
                wav_id=f"{prefix}_st_{iso_src}2{iso_tgt}_{event_id}",
                wav_path=path_template.format(str(audio_path.resolve())),
                start_time=float(r["start_time"]),
                end_time=float(r["end_time"]),
                lang=f"<{iso_src}>",
                task=f"<st_{iso_tgt}>",
                text=tgt_text,
                asr_text=src_text,
            )
        )

    return list(talks.values())


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
        default=["train", "dev", "test"],
        help="Data splits to prepare.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

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

        all_talks = []

        # ASR part
        for lang in asr_langs:
            talks = collect_data_asr(
                data_dir=args.data_dir,
                split=split,
                prefix=args.prefix,
                src=lang,
            )
            # Note: The lang-id here is from original VoxPopuli
            # We will transform it into ISO-639-3 later
            logging.info(f"ASR: Split={split} | Lang={lang} | NumTalk={len(talks)}")
            all_talks = all_talks + talks

        # ST part
        for src in st_src_langs:
            for tgt in st_tgt_langs_human_transcription:
                if src == tgt:
                    continue
                talks = collect_data_st(
                    data_dir=args.data_dir,
                    split=split,
                    prefix=args.prefix,
                    src=src,
                    tgt=tgt,
                )
                # Note: The lang-id here is from original VoxPopuli
                # We will transform it into ISO-639-3 later
                logging.info(
                    f"ST: Split={split} | Lang={src}2{tgt} | NumTalk={len(talks)}"
                )
                all_talks = all_talks + talks

        for talk in all_talks:
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
        *[f"<{lang}>" for lang in set(asr_langs + st_src_langs)],
        *[f"<st_{lang}>" for lang in st_tgt_langs_human_transcription],
        *SYMBOLS_TIME,
    ]
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")
