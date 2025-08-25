"""
This script concatenates consecutive segments of audio files into long-form utterances,
which can then be used for Whisper/OWSM-style training.
Reference: Section 2.1.1 in the paper (https://arxiv.org/pdf/2506.00338)
"""

import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from utils import (
    TO_ISO_LANGUAGE_CODE,
    LongUtterance,
    Utterance,
    generate_long_utterances,
)


def construct_data_from_file(file):
    lang = Path(file).resolve().parent.parent.name[:2]  # two-letter language code
    if lang == "iw":
        lang = "he"
    lang = TO_ISO_LANGUAGE_CODE[lang]  # convert to three-letter language code

    long_utts = []
    with open(file, "r") as f:
        for line in f:
            recording = json.loads(line.strip())

            short_utts = []
            audio_id = recording["audio_id"]
            wav_path = recording["wav_path"]

            for (
                utt_id,
                start_time,
                end_time,
                confidence,
                cleaned_text,
                raw_text,
            ) in recording["utts"]:
                short_utts.append(
                    Utterance(
                        utt_id=utt_id,
                        wav_id=audio_id,
                        wav_path=wav_path,
                        start_time=start_time,
                        end_time=end_time,
                        lang=f"<{lang}>",
                        task="<asr>",
                        text=cleaned_text,
                        asr_text=cleaned_text,
                        confidence=confidence,
                    )
                )

            long_utts.extend(generate_long_utterances(short_utts))

    return long_utts


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("-d", "--yodas2_dir", type=str, required=True)
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="Path to save the long-form data file",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    yodas2_dir = Path(args.yodas2_dir)
    assert yodas2_dir.is_dir(), f"YODAS2 path {yodas2_dir} is not a directory."

    all_files = list(yodas2_dir.glob("data/*/text_reseg/*.jsonl"))
    with open(args.output_file, "w") as fout:
        for file in tqdm(all_files):
            long_utts = construct_data_from_file(file)
            for long_utt in long_utts:
                fout.write(json.dumps(long_utt.__dict__, ensure_ascii=False) + "\n")
