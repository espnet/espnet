"""Prepare TEDLIUM data for English ASR."""

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


# Copied from https://huggingface.co/datasets/LIUM/tedlium/blob/main/tedlium.py
def _maybe_trim_suffix(transcript):
    # stm files for the TEDLIUM release 1 train split contain a key (enclosed in
    # parens) at the end.
    splits = transcript.rsplit(" ", 1)
    transcript = splits[0]
    if len(splits) > 1:
        suffix = splits[-1]
        if not suffix.startswith("("):
            transcript += " " + suffix
    return transcript


def collect_data(
    data_dir: Union[Path, str], split: str, prefix: str
) -> List[List[Utterance]]:
    """Collect utterances in each long talk."""
    sph_dir = Path(data_dir) / split / "sph"
    stm_dir = Path(data_dir) / split / "stm"

    ret = []
    for stm in stm_dir.iterdir():
        sph = sph_dir / (stm.name.removesuffix(".stm") + ".sph")
        assert sph.is_file(), sph

        utts = []
        with open(stm, "r") as fp:
            for line in fp.readlines():
                line = line.strip()
                (
                    filename,
                    channel,
                    speaker,
                    start_time,
                    end_time,
                    label,
                    transcript,
                ) = line.split(" ", 6)
                if "ignore_time_segment_in_scoring" not in transcript:
                    transcript = _maybe_trim_suffix(transcript)
                    transcript = transcript.replace("<unk>", "")
                    transcript = " ".join(transcript.split())
                    if transcript:
                        utts.append(
                            Utterance(
                                utt_id=(
                                    f"{prefix}_{filename}_"
                                    f"{float(start_time) * 1000:09.0f}_"
                                    f"{float(end_time) * 1000:09.0f}"
                                ),
                                wav_id=f"{prefix}_{filename}",
                                wav_path=f"sph2pipe -f wav -p {sph} |",
                                start_time=float(start_time),
                                end_time=float(end_time),
                                lang="<en>",
                                task="<asr>",
                                text=transcript,
                                asr_text=transcript,
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
