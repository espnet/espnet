"""Prepare AISHELL data for Chinese ASR."""

from argparse import ArgumentParser
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


def collect_data(
    data_dir: Union[Path, str], split: str, prefix: str
) -> List[List[Utterance]]:
    """Collect utterances."""
    data_dir = Path(data_dir)

    name2text = {}
    with open(
        data_dir / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt", "r"
    ) as fp:
        for line in fp.readlines():
            name, text = line.split(maxsplit=1)
            name2text[name] = "".join(text.split())

    ret = []
    for wav_file in (data_dir / "data_aishell" / "wav" / split).glob("*/*.wav"):
        if wav_file.name.removesuffix(".wav") in name2text:
            text = name2text[wav_file.name.removesuffix(".wav")]
            ret.append(
                [
                    Utterance(
                        utt_id=f"{prefix}_{wav_file.name.removesuffix('.wav')}",
                        wav_id=f"{prefix}_{wav_file.name.removesuffix('.wav')}",
                        wav_path=str(wav_file.resolve()),
                        start_time=0.0,
                        end_time=librosa.get_duration(filename=wav_file),
                        lang="<zh>",
                        task="<asr>",
                        text=text,
                        asr_text=text,
                    )
                ]
            )
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
        "<zh>",
        "<asr>",
        *SYMBOLS_TIME,
    ]
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")
