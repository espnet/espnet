#!/usr/bin/env python3
import argparse
from typing import Type

from espnet2.main_funcs.pack_funcs import pack


class PackedContents:
    files = []
    yaml_files = []


class ASRPackedContents(PackedContents):
    # These names must be consistent with the argument of inference functions
    files = ["asr_model_file", "lm_file"]
    yaml_files = ["asr_train_config", "lm_train_config"]


class STPackedContents(PackedContents):
    files = ["st_model_file"]
    yaml_files = ["st_train_config"]


class TTSPackedContents(PackedContents):
    files = ["model_file"]
    yaml_files = ["train_config"]


class EnhPackedContents(PackedContents):
    files = ["model_file"]
    yaml_files = ["train_config"]


class DiarPackedContents(PackedContents):
    files = ["model_file"]
    yaml_files = ["train_config"]


class SVSPackedContents(PackedContents):
    files = ["model_file"]
    yaml_files = ["train_config"]


class EnhS2TPackedContents(PackedContents):
    # These names must be consistent with the argument of inference functions
    files = ["enh_s2t_model_file", "lm_file"]
    yaml_files = ["enh_s2t_train_config", "lm_train_config"]


class SSLPackedContents(PackedContents):
    # These names must be consistent with the argument of inference functions
    files = ["model_file"]
    yaml_files = ["train_config"]


class S2STPackedContents(PackedContents):
    # These names must be consistent with the argument of inference functions
    files = ["s2st_model_file"]
    yaml_files = ["s2st_train_config"]


class S2TPackedContents(PackedContents):
    # These names must be consistent with the argument of inference functions
    files = ["s2t_model_file", "lm_file"]
    yaml_files = ["s2t_train_config", "lm_train_config"]


class ClassificationPackedContents(PackedContents):
    # These names must be consistent with the argument of inference functions
    files = ["classification_model_file"]
    yaml_files = ["classification_train_config"]


class SpkPackedContents(PackedContents):
    files = ["model_file"]
    yaml_files = ["train_config"]


class CodecPackedContents(PackedContents):
    files = ["model_file"]
    yaml_files = ["train_config"]


def add_arguments(parser: argparse.ArgumentParser, contents: Type[PackedContents]):
    parser.add_argument("--outpath", type=str, required=True)
    for key in contents.yaml_files:
        parser.add_argument(f"--{key}", type=str, default=None)
    for key in contents.files:
        parser.add_argument(f"--{key}", type=str, default=None)
    parser.add_argument("--option", type=str, action="append", default=[])


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack input files to archive format")
    subparsers = parser.add_subparsers()

    # Create subparser for different tasks
    for name, contents in [
        ("asr", ASRPackedContents),
        ("st", STPackedContents),
        ("tts", TTSPackedContents),
        ("enh", EnhPackedContents),
        ("diar", DiarPackedContents),
        ("svs", SVSPackedContents),
        ("enh_s2t", EnhS2TPackedContents),
        ("ssl", SSLPackedContents),
        ("s2st", S2STPackedContents),
        ("s2t", S2TPackedContents),
        ("spk", SpkPackedContents),
        ("codec", CodecPackedContents),
        ("cls", ClassificationPackedContents),
    ]:
        parser_asr = subparsers.add_parser(
            name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    )


if __name__ == "__main__":
    main()
