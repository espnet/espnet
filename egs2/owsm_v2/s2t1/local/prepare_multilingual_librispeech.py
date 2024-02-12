"""Prepare Multilingual LibriSpeech data for ASR."""

import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests

from utils import (
    LANGUAGES,
    SYMBOL_NA,
    SYMBOL_NOSPEECH,
    SYMBOLS_TIME,
    LongUtterance,
    Utterance,
    generate_long_utterances,
)


def request_with_retry(url, n_retries=3, success_list=[200], **kwargs):
    for _ in range(n_retries):
        try:
            response = requests.get(url, **kwargs)
            if response.status_code in success_list:
                # Return response if successful
                return response
        except Exception as e:
            print(e)
            pass
    return None


def collect_data(
    data_dir: Union[Path, str], lang: str, split: str, prefix: str
) -> List[List[Utterance]]:
    """Collect utterances in each long talk."""
    data_dir = Path(data_dir) / f"mls_{LANGUAGES[lang]}" / split
    download_dir = data_dir / "audio_long"  # download long audios
    download_dir.mkdir(parents=True, exist_ok=True)

    uttid2trans = {}
    with open(data_dir / "transcripts.txt", "r") as fp:
        for line in fp:
            uttid, trans = line.strip().split(maxsplit=1)
            uttid2trans[uttid] = " ".join(trans.split())

    url2utts = defaultdict(list)
    with open(data_dir / "segments.txt", "r") as fp:
        for line in fp:
            uttid, url, start_time, end_time = line.strip().split()
            url2utts[url].append(
                (uttid, float(start_time), float(end_time), uttid2trans[uttid])
            )

    ret = []
    for url, utts in url2utts.items():
        mp3_file = download_dir / url.split("/")[-1]
        if mp3_file.is_file():
            print(f"Skip downloading {mp3_file}")
        else:
            response = request_with_retry(url)
            if response is None:
                print(f"Failed to download {mp3_file}. Skip it.")
                continue
            with open(mp3_file, "wb") as fp:
                fp.write(response.content)

        short_utts = []
        for uttid, start_time, end_time, trans in utts:
            short_utts.append(
                Utterance(
                    utt_id=f"{prefix}_{lang}_{uttid}",
                    wav_id=f"{prefix}_{lang}_{url.split('/')[-1].removesuffix('.mp3')}",
                    wav_path=(
                        f"ffmpeg -i {str(mp3_file.resolve())} -ac 1 -ar 16000 -f"
                        " wav - |"
                    ),
                    start_time=start_time,
                    end_time=end_time,
                    lang=f"<{lang}>",
                    task="<asr>",
                    text=trans,
                    asr_text=trans,
                )
            )
        ret.append(short_utts)
    return ret


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
        default=["dev", "train"],
        help="Data splits to prepare.",
    )
    parser.add_argument(
        "--langs",
        type=str,
        nargs="+",
        default=["nl", "en", "fr", "de", "it", "pl", "pt", "es"],
        help="ASR languages.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = [
        SYMBOL_NA,
        SYMBOL_NOSPEECH,
        *[f"<{x}>" for x in args.langs],
        "<asr>",
        *SYMBOLS_TIME,
    ]
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")

    for lang in args.langs:
        for split in args.splits:
            write_dir = args.output_dir / f"{split}.{lang}"
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
                lang=lang,
                split=split,
                prefix=args.prefix,
            )
            print(f"Found {len(talks)} long audios for {split}.{lang}")
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
