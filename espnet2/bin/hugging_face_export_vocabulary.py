#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import List

from typeguard import typechecked

from espnet.utils.cli_utils import get_commandline_args

try:
    from transformers import AutoTokenizer

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


@typechecked
def export_vocabulary(
    output: str,
    model_name_or_path: str,
    log_level: str,
    add_symbol: List[str],
):

    if not is_transformers_available:
        raise ImportError(
            "`transformers` is not available. Please install it via `pip install"
            " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
            " && ./installers/install_transformers.sh`."
        )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    if output == "-":
        fout = sys.stdout
    else:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        fout = p.open("w", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    words = ["" for _ in range(tokenizer.vocab_size)]
    vocab = tokenizer.get_vocab()

    for w in vocab:
        if vocab[w] < tokenizer.vocab_size:  # pythia tokenizer
            words[vocab[w]] = w

    # Parse the values of --add_symbol
    for symbol_and_id in add_symbol:
        # e.g symbol="<blank>:0"
        try:
            symbol, idx = symbol_and_id.split(":")
            idx = int(idx)
        except ValueError:
            raise RuntimeError(f"Format error: e.g. '<blank>:0': {symbol_and_id}")
        symbol = symbol.strip()

        # e.g. idx=0  -> append as the first symbol
        # e.g. idx=-1 -> append as the last symbol
        if idx < 0:
            idx = len(words) + 1 + idx
        words.insert(idx, symbol)

    # Write words
    for w in words:
        fout.write(w + "\n")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Hugging Face vocabulary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output", "-o", required=True, help="Output text. - indicates sys.stdout"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Hugging Face model name or path",
    )

    parser.add_argument(
        "--add_symbol",
        type=str,
        default=[],
        action="append",
        help="Append symbol e.g. --add_symbol '<blank>:0' --add_symbol '<unk>:1'",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    export_vocabulary(**kwargs)


if __name__ == "__main__":
    main()
