"""
This script converts the filtered data into Kaldi format for later training.
"""

import json
from argparse import ArgumentParser
from pathlib import Path


def is_valid(text: str) -> bool:
    invalid_syms = ["<s>", "</s>", "#0"]
    for sym in invalid_syms:
        if text.find(sym) != -1:
            return False
    return True


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, required=True, help="Input directory path"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Output directory path"
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    outroot = args.out_dir
    Path(outroot).mkdir(parents=True, exist_ok=True)

    f_text = open(f"{outroot}/text", "w")
    f_textctc = open(f"{outroot}/text.ctc", "w")
    f_textprev = open(f"{outroot}/text.prev", "w")
    f_utt2spk = open(f"{outroot}/utt2spk", "w")
    f_wavscp = open(f"{outroot}/wav.scp", "w")
    f_segments = open(f"{outroot}/segments", "w")

    for file in Path(args.in_dir).glob(f"*/quantile_0.10.jsonl"):
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line.strip())
                utt_id = sample["utt_id"]
                wav_id = sample["wav_id"]

                if (
                    is_valid(sample["text_with_time"])
                    and is_valid(sample["asr_text"])
                    and is_valid(sample["prev_text"])
                ):
                    f_text.write(
                        f"{utt_id} {sample['lang']}{sample['task']}"
                        f"{sample['text_with_time']}\n"
                    )
                    f_textctc.write(f"{utt_id} {sample['asr_text']}\n")
                    f_textprev.write(f"{utt_id} {sample['prev_text']}\n")
                    f_utt2spk.write(f"{utt_id} {utt_id}\n")
                    f_segments.write(
                        f"{utt_id} {wav_id} {sample['start_time']} "
                        f"{sample['end_time']}\n"
                    )
                    f_wavscp.write(
                        f"{wav_id} sox {sample['wav_path']} -t wav -r 16k -c 1 - |\n"
                    )
                else:
                    print(f"Invalid: {utt_id}")
                    print(f"Text with time: {sample['text_with_time']}")
                    print(f"ASR text: {sample['asr_text']}")
                    print(f"Prev text: {sample['prev_text']}")

    f_text.close()
    f_textctc.close()
    f_textprev.close()
    f_utt2spk.close()
    f_wavscp.close()
    f_segments.close()
