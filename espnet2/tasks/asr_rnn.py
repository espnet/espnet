import argparse
from typing import Any, Dict

import configargparse
from typeguard import typechecked

from espnet2.tasks.base_task import BaseTask


class ASRRNNTask(BaseTask):
    @classmethod
    @typechecked
    def add_arguments(cls, parser: configargparse.ArgumentParser = None) \
            -> configargparse.ArgumentParser:
        # Note(kamo): Use '_' instead of '-' to avoid confusion as separator
        if parser is None:
            parser = configargparse.ArgumentParser(
                description='',
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
        BaseTask.add_arguments(parser)
        return parser

    @classmethod
    @typechecked
    def get_default_config(cls) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace):
        # TODO(kamo): Create Encoder and Decoder class to fit the interface.
        raise NotImplementedError


if __name__ == '__main__':
    ASRRNNTask.main()
