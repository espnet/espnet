import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from espnet.utils.cli_utils import get_commandline_args
from espnet2.text.tokenizer import build_tokenizer
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none


def tokenize(
    input: str,
    output: str,
    token_type: str,
    space_symbol: str,
    delimiter: Optional[str],
    non_language_symbols: Optional[str],
    bpemodel: Optional[str],
    log_level: str,
    write_vocabulary: bool,
    vocabulary_size: int,
    remove_non_language_symbols: bool,
    cutoff: int,
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
        remove_non_language_symbols=remove_non_language_symbols,
    )

    if write_vocabulary:
        counter = Counter()

    for line in fin:
        line = line.rstrip()
        tokens = tokenizer.text2tokens(line)
        if not write_vocabulary:
            fout.write(" ".join(tokens) + "\n")
        else:
            for t in tokens:
                counter[t] += 1

    if not write_vocabulary:
        return

    total_count = sum(counter.values())
    invocab_count = 0
    for nvocab, (w, c) in enumerate(sorted(counter.items(), key=lambda x: x[1]), 1):
        if c <= cutoff:
            break
        if nvocab >= vocabulary_size > 0:
            break
        fout.write(w + "\n")
        invocab_count += c

    logging.info(
        f"OOV rate = {float(total_count - invocab_count) / total_count * 100} %")


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
    parser.add_argument("--token_type", "-t", default="char",
                        choices=["char", "bpe", "word"], help="Token type")
    parser.add_argument("--delimiter", "-d", default=None,
                        help="The delimiter")
    parser.add_argument("--space_symbol", default="<space>",
                        help="The space symbol")
    parser.add_argument("--bpemodel", default=None,
                        help="The bpemodel file path")
    parser.add_argument("--non_language_symbols", type=str_or_none,
                        help="non_language_symbols file path")
    parser.add_argument("--remove_non_language_symbols",
                        type=str2bool, default=False,
                        help="Remove non-language-symbols from tokens")
    parser.add_argument("--write_vocabulary",
                        type=str2bool, default=False,
                        help="Write tokens list instead of tokenized text per line")
    parser.add_argument("--vocabulary_size", type=int, default=0,
                        help="Vocabulary size")
    parser.add_argument("--cutoff", default=0, type=int,
                        help="cut-off frequency used for write-vocabulary mode")
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    tokenize(**kwargs)


if __name__ == "__main__":
    main()
