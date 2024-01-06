from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from utils import SYMBOL_NA, SYMBOL_NOSPEECH, SYMBOLS_TIME, TO_ISO_LANGUAGE_CODE


def filter_lang_id(reader, writer):
    langs, st_langs = set(), set()
    for line in reader:
        utt, ctx = line.strip().split(maxsplit=1)
        src, tgt, other = ctx.split("><", maxsplit=2)

        src = TO_ISO_LANGUAGE_CODE[src[1:]]
        langs.add(src)

        tgt = (
            tgt
            if tgt == "asr"
            else "st_" + TO_ISO_LANGUAGE_CODE[tgt.replace("st_", "")]
        )
        if tgt.startswith("st_"):
            st_langs.add(tgt.replace("st_", ""))

        line = f"{utt} <{src}><{tgt}><{other}\n"
        writer.write(line)

    langs = list(langs)
    langs.sort()

    st_langs = list(st_langs)
    st_langs.sort()

    return langs, st_langs


def get_parser():
    parser = ArgumentParser(description="Show statistics of the data directory.")
    parser.add_argument(
        "-i", "--input", type=Path, required=True, help="input text file"
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="output text file"
    )
    parser.add_argument(
        "--nlsyms", type=Path, default=None, help="output path of nlsyms"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    writer = open(args.output, "w")
    reader = open(args.input, "r")
    langs, st_langs = filter_lang_id(reader, writer)
    writer.close()
    reader.close()

    if args.nlsyms is not None:
        special_tokens = [
            SYMBOL_NA,
            SYMBOL_NOSPEECH,
            *[f"<{lang}>" for lang in langs],
            *[f"<st_{lang}>" for lang in st_langs],
            *SYMBOLS_TIME,
        ]

        with open(args.nlsyms, "w") as fp:
            for tok in special_tokens:
                fp.write(f"{tok}\n")
