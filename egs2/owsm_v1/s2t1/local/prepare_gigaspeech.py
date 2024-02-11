"""Prepare GigaSpeech data for English ASR."""

import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from utils import (
    SYMBOL_NA,
    SYMBOL_NOSPEECH,
    SYMBOLS_TIME,
    LongUtterance,
    Utterance,
    generate_long_utterances,
)


def preprocess_text(text: str) -> str:
    garbage_tags = [
        "<SIL>",
        "<MUSIC>",
        "<NOISE>",
        "<OTHER>",
    ]
    for g in garbage_tags:
        if g in text:
            return ""

    text = text.replace(" <COMMA>", ",")
    text = text.replace(" <PERIOD>", ".")
    text = text.replace(" <QUESTIONMARK>", "?")
    text = text.replace(" <EXCLAMATIONPOINT>", "!")

    assert "<" not in text and ">" not in text, text
    return text.lower()


def collect_data(
    data_dir: Union[Path, str], split: str, prefix: str
) -> List[List[Utterance]]:
    """Collect utterances in each long talk."""
    data_dir = Path(data_dir)
    with open(data_dir / "GigaSpeech.json", "r") as fp:
        info = json.load(fp)

    ret = []
    for audio in info["audios"]:
        if ("{" + split + "}") in audio["subsets"]:
            wav_path = data_dir / audio["path"]
            assert wav_path.is_file()

            utts = []
            for seg in audio["segments"]:
                text = preprocess_text(seg["text_tn"])
                if text:
                    utts.append(
                        Utterance(
                            utt_id=f"{prefix}_{seg['sid']}",
                            wav_id=f"{prefix}_{audio['aid']}",
                            wav_path=(
                                f"ffmpeg -i {str(wav_path.resolve())} -ac 1 -ar 16000"
                                " -f wav - |"
                            ),
                            start_time=seg["begin_time"],
                            end_time=seg["end_time"],
                            lang="<en>",
                            task="<asr>",
                            text=text,
                            asr_text=text,
                        )
                    )
            ret.append(utts)
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
        "<en>",
        "<asr>",
        *SYMBOLS_TIME,
    ]
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")
