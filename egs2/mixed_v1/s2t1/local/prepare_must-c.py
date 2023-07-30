"""Prepare MuST-C v1, v2, or v3.

MuST-C contains data of two tasks:
1. En ASR
2. En->X ST

"""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml

from utils import (
    SYMBOL_NA,
    SYMBOL_NOSPEECH,
    SYMBOLS_TIME,
    LongUtterance,
    Utterance,
    generate_long_utterances,
)


def collect_data(
    data_dir: Union[Path, str], lang: str, split: str, prefix: str
) -> List[List[Utterance]]:
    """Collect utterances in each long talk."""

    data_dir = Path(data_dir)
    txt_dir = data_dir / f"en-{lang}" / "data" / split / "txt"
    wav_dir = data_dir / f"en-{lang}" / "data" / split / "wav"

    with open(txt_dir / f"{split}.yaml", "r") as fp:
        utts = yaml.safe_load(fp)

    with open(txt_dir / f"{split}.en", "r") as fp:
        src_text = [ln.strip() for ln in fp.readlines()]

    with open(txt_dir / f"{split}.{lang}", "r") as fp:
        tgt_text = [ln.strip() for ln in fp.readlines()]

    assert len(src_text) == len(tgt_text) and len(tgt_text) == len(utts)

    wav2utts = defaultdict(list)
    for utt, src, tgt in zip(utts, src_text, tgt_text):
        wav_name: str = utt["wav"]  # e.g.: 'ted_767.wav'
        wav_path = str((wav_dir / wav_name).resolve())
        wav_id = f"{prefix}_{wav_name.removesuffix('.wav')}"
        start_time = float(utt["offset"])
        end_time = start_time + float(utt["duration"])

        # NOTE(yifan): each utterance can be used in two tasks (asr, st)
        wav2utts[wav_name + ".asr"].append(
            Utterance(
                utt_id=(
                    f"{wav_id}_{round(1000 * start_time):07d}"
                    f"_{round(1000 * end_time):07d}_asr"
                ),
                wav_id=wav_id,
                wav_path=wav_path,
                start_time=start_time,
                end_time=end_time,
                lang="<en>",
                task="<asr>",
                text=" ".join(src.split()),
                asr_text=" ".join(src.split()),
            )
        )
        wav2utts[wav_name + f".st_{lang}"].append(
            Utterance(
                utt_id=(
                    f"{wav_id}_{round(1000 * start_time):07d}"
                    f"_{round(1000 * end_time):07d}_st_{lang}"
                ),
                wav_id=wav_id,
                wav_path=wav_path,
                start_time=start_time,
                end_time=end_time,
                lang="<en>",
                task=f"<st_{lang}>",
                text=" ".join(tgt.split()),
                asr_text=" ".join(src.split()),
            )
        )

    return list(wav2utts.values())


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
        "--languages",
        type=str,
        nargs="+",
        default=[],
        help="Target languages that will be prepared.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if len(args.languages) > 0:
        languages = args.languages
    else:
        languages = [
            d.name.removeprefix("en-") for d in args.data_dir.iterdir() if d.is_dir()
        ]

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

        for lang in languages:
            talks = collect_data(
                data_dir=args.data_dir,
                lang=lang,
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
        *[f"<st_{x}>" for x in languages],
        *SYMBOLS_TIME,
    ]
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")
