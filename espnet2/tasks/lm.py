import argparse
from typing import Any, Dict, Type, Tuple, Optional

import configargparse
import torch
from typeguard import typechecked

from espnet2.lm.model import Model
from espnet2.lm.seq_rnn import SequentialRNNLM
from espnet2.tasks.base_task import BaseTask
from espnet2.utils.get_default_values import get_defaut_values
from espnet2.utils.types import str_or_none, NestedDictAction


class LMTask(BaseTask):
    @classmethod
    @typechecked
    def add_arguments(cls, parser: configargparse.ArgumentParser = None) \
            -> configargparse.ArgumentParser:
        # Note(kamo): Use '_' instead of '-' to avoid confusion as separator
        if parser is None:
            parser = configargparse.ArgumentParser(
                description='Train launguage model',
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

        BaseTask.add_arguments(parser)
        group = parser.add_argument_group(description='Task related')

        # Note(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default('required')
        required += []

        group.add_argument(
            '--lm', type=str_or_none, default='seq_rnn',
            choices=cls.lm_choices(), help='Specify frontend class')
        group.add_argument(
            '--lm_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for lm class.')
        group.add_argument(
            '--model_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for Model class.')

        return parser

    @classmethod
    @typechecked
    def exclude_opts(cls) -> Tuple[str, ...]:
        return BaseTask.exclude_opts()

    @classmethod
    @typechecked
    def get_default_config(cls) -> Dict[str, Any]:
        # 0. Parse command line arguments
        parser = LMTask.add_arguments()
        args, _ = parser.parse_known_args()

        # 1. Get the default values from class.__init__
        lm_class = cls.get_lm_class(args.lm)
        lm_conf = get_defaut_values(lm_class)
        model_conf = get_defaut_values(Model)

        # 2. Create configuration-dict from command-arguments
        config = vars(args)

        # 3. Update the dict using the inherited configuration from BaseTask
        config.update(BaseTask.get_default_config())

        # 4. Overwrite the default config by the command-arguments
        lm_conf.update(config['lm_conf'])
        model_conf.update(config['model_conf'])

        # 5. Reassign them to the configuration
        config.update(lm_conf=lm_conf, model_conf=model_conf)

        # 6. Excludes the specified options
        for k in cls.exclude_opts():
            config.pop(k)

        return config

    @classmethod
    @typechecked
    def lm_choices(cls) -> Tuple[Optional[str], ...]:
        choices = ('seq_rnn',)
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        return choices

    @classmethod
    @typechecked
    def get_lm_class(cls, name: str) -> Type[torch.nn.Module]:
        if name.lower() == 'seq_rnn':
            return SequentialRNNLM
        else:
            raise RuntimeError(
                f'--frontend must be one of '
                f'{cls.lm_choices()}: --frontend {name}')

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> torch.nn.Module:
        lm_class = cls.get_lm_class()
        lm = lm_class(**args.lm_conf)
        model = Model(lm=lm, **args.model_conf)
        return model


if __name__ == '__main__':
    # These two are equivalent:
    #   % python -m espnet2.tasks.lm
    #   % python -m espnet2.bin.train lm

    import sys
    from espnet.utils.cli_utils import get_commandline_args
    print(get_commandline_args(), file=sys.stderr)
    LMTask.main()

