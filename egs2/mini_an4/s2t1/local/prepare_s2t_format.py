from argparse import ArgumentParser
from pathlib import Path

from kaldiio import ReadHelper

SYMBOL_NA: str = "<na>"  # symbol denoting text is not available
SPEECH_RESOLUTION: float = 0.02  # resolution in seconds


def time2token(x: float) -> str:
    """Convert float time to timestamp token."""
    x = round(x / SPEECH_RESOLUTION) * SPEECH_RESOLUTION
    return f"<{x:.2f}>"


def get_parser():
    parser = ArgumentParser(description="Convert text to s2t format.")
    parser.add_argument("--data_dir", type=Path, help="Data directory")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    utt2dur = {}
    with ReadHelper(f'scp:{str(args.data_dir / "wav.scp")}') as reader:
        for key, (rate, numpy_array) in reader:
            utt2dur[key] = numpy_array.shape[-1] / rate

    with (args.data_dir / "text.ctc").open("r") as f_text_ctc, (
        args.data_dir / "text"
    ).open("w") as f_text, (args.data_dir / "text.prev").open("w") as f_text_prev:
        for line in f_text_ctc:
            uttid, raw_text = line.strip().split(maxsplit=1)
            f_text.write(
                f"{uttid} <eng><asr><0.00> "
                f"{raw_text.strip()}{time2token(utt2dur[uttid])}\n"
            )
            f_text_prev.write(f"{uttid} {SYMBOL_NA}\n")
