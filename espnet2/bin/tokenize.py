import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from espnet.utils.cli_utils import get_commandline_args
from espnet2.text.tokenizer import build_tokenizer


def tokenize(
    input: str,
    output: str,
    token_type: str,
    space_symbol: str,
    delimiter: Optional[str],
    non_language_symbols: Optional[str],
    bpemodel: Optional[str],
    log_level: str,
):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    if input == "-":
        fin = sys.stdin
    else:
        fin = Path(input).open("r")
    if output == "-":
        fout = sys.stdout
    else:
        p = Path(output)
        p.parent.mkdir(parents=True, exists_ok=True)
        fout = p.open("w")

    tokenizer = build_tokenizer(
        token_type=token_type,
        bpemodel=bpemodel,
        delimiter=delimiter,
        space_symbol=space_symbol,
        non_language_symbols=non_language_symbols,
    )
    for line in fin:
        line = line.rstrip()
        tokens = tokenizer.text2tokens(line)
        fout.write(" ".join(tokens) + "\n")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenize texts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("INFO", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--input", "-i", required=True,
                        help="Input text. - indicates sys.stdin")
    parser.add_argument("--output", "-o", required=True,
                        help="Output text. - indicates sys.stdout")
    parser.add_argument("--token-type", "-t", default="char",
                        choices=["char", "bpe", "word"], help="Token type")
    parser.add_argument("--delimiter", "-d", default=None,
                        help="The delimiter")
    parser.add_argument("--space-symbol", default="<space>",
                        help="The space symbol")
    parser.add_argument("--bpemodel", default=None,
                        help="The bpemodel file path")
    parser.add_argument("--non-language-symbols", default=None,
                        help="non_language_symbols file path")
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    tokenize(**kwargs)


if __name__ == "__main__":
    main()
