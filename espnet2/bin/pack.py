import argparse
from typing import Type

from espnet2.utils.pack_funcs import pack


class PackedContents:
    files = []
    yaml_files = []


class ASRPackedContents(PackedContents):
    files = ["asr_model_file.pth", "lm_file.pth"]
    yaml_files = ["asr_train_config.yaml", "lm_train_config.yaml"]


class TTSPackedContents(PackedContents):
    files = ["model_file.pth"]
    yaml_files = ["train_config.yaml"]


def add_arguments(parser: argparse.ArgumentParser, contents: Type[PackedContents]):
    parser.add_argument("--outpath", type=str, required=True)
    for key in contents.yaml_files:
        parser.add_argument(f"--{key}", type=str, default=None)
    for key in contents.files:
        parser.add_argument(f"--{key}", type=str, default=None)
    parser.add_argument("--option", type=str, action="append", default=[])
    parser.add_argument(
        "--mode",
        type=str,
        default="w:gz",
        choices=["w", "w:gz", "w:bz2", "w:xz"],
        help="Compression mode",
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pack input files to archive format. If the external file path "
        "are written in the input yaml files, then the paths are "
        "rewritten to the archived name",
    )
    subparsers = parser.add_subparsers()

    # Create subparser for ASR
    for name, contents in [("asr", ASRPackedContents), ("tts", TTSPackedContents)]:
        parser_asr = subparsers.add_parser(
            name, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        add_arguments(parser_asr, contents)
        parser_asr.set_defaults(contents=contents)
    return parser


def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)
    if not hasattr(args, "contents"):
        parser.print_help()
        parser.exit(2)

    yaml_files = {
        y: getattr(args, y)
        for y in args.contents.yaml_files
        if getattr(args, y) is not None
    }
    files = {
        y: getattr(args, y) for y in args.contents.files if getattr(args, y) is not None
    }
    pack(
        yaml_files=yaml_files,
        files=files,
        option=args.option,
        outpath=args.outpath,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
