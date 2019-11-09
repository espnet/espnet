import argparse
from typing import Any, Dict

import configargparse
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.ctc import CTC
from espnet2.asr.e2e import E2E
from espnet2.train.base_task import BaseTask


class ASRRNNTask(BaseTask):
    @classmethod
    def add_arguments(cls, parser: configargparse.ArgumentParser = None) \
            -> configargparse.ArgumentParser:
        assert check_argument_types()
        # Note1(kamo): Use '_' instead of '-' to avoid confusion as separator
        # Note2(kamo): Any required arguments can't be used
        # to provide --show_config mode.
        # Instead of it, insert checking if it is given like:
        #   >>> cls.check_required(args, 'output_dir')
        if parser is None:
            parser = configargparse.ArgumentParser(
                description='',
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
        BaseTask.add_arguments(parser)
        return parser

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def build_model(cls, args: argparse.Namespace):
        assert check_argument_types()
        # TODO(kamo): Create Encoder and Decoder class to fit the interface.
        raise NotImplementedError


if __name__ == '__main__':
    ASRRNNTask.main()
