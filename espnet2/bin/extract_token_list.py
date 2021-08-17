import argparse
from pathlib import Path
from typeguard import check_argument_types
from typing import Union

import yaml

from espnet2.utils import config_argparse


def extract_token_list(
    config_file: Union[Path, str], token_file: Union[Path, str]
) -> None:
    """This method is used to extract tokens.txt from config_file

    Args:
        config_file: The yaml file saved when training.
        token_file: Path to save token_list for compiling TLG later
    """
    assert check_argument_types()
    config_file = Path(config_file)
    token_file = Path(token_file)
    with config_file.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
        assert "token_list" in args
        token_list = args["token_list"]
        token_and_idx_lines = ""
        for token_idx, token in enumerate(token_list):
            token_and_idx_lines += f"{token} {token_idx}\n"

        token_file.write_text(token_and_idx_lines)


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Extract token list",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--token_file", type=str, required=True)
    return parser


def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    extract_token_list(**kwargs)


if __name__ == "__main__":
    main()
