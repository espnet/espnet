import argparse
import logging
from typing import Any, Dict, Type, Tuple, Optional

import configargparse
from typeguard import typechecked

from espnet2.tasks.base_task import AbsTask
from espnet2.tts.abs_model import AbsTTSModel
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.utils.get_default_kwargs import get_defaut_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str_or_none


class TTSTask(AbsTask):
    @classmethod
    @typechecked
    def add_arguments(cls, parser: configargparse.ArgumentParser = None) \
            -> configargparse.ArgumentParser:
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        if parser is None:
            parser = configargparse.ArgumentParser(
                description='Train launguage model',
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

        AbsTask.add_arguments(parser)
        group = parser.add_argument_group(description='Task related')

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default('required')
        required += ['token_list', 'odim']

        group.add_argument('--odim', type=int, default=None,
                           help='The number of output dimension of the feature')
        group.add_argument('--token_list', type=str_or_none, default=None,
                           help='A text mapping int-id to token')
        group.add_argument(
            '--model', type=str, default='tacotron2',
            choices=cls.model_choices(), help='Specify frontend class')
        group.add_argument(
            '--model_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for Model class.')

        return parser

    @classmethod
    @typechecked
    def exclude_opts(cls) -> Tuple[str, ...]:
        """The options not to be shown by --print_config"""
        return AbsTask.exclude_opts()

    @classmethod
    @typechecked
    def get_default_config(cls) -> Dict[str, Any]:
        # This method is used only for --print_config

        # 0. Parse command line arguments
        parser = TTSTask.add_arguments()
        args, _ = parser.parse_known_args()

        # 1. Get the default values from class.__init__
        model_class = cls.get_model_class(args.model)
        model_conf = get_defaut_kwargs(model_class)

        # 2. Create configuration-dict from command-arguments
        config = vars(args)

        # 3. Update the dict using the inherited configuration from BaseTask
        config.update(AbsTask.get_default_config())

        # 4. Overwrite the default config by the command-arguments
        model_conf.update(config['model_conf'])

        # 5. Reassign them to the configuration
        config.update(model_conf=model_conf)

        # 6. Excludes the options not to be shown
        for k in cls.exclude_opts():
            config.pop(k)

        return config

    @classmethod
    @typechecked
    def model_choices(cls) -> Tuple[Optional[str], ...]:
        choices = ('Tacotron2',)
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        return choices

    @classmethod
    @typechecked
    def get_model_class(cls, name: str) -> Type[AbsTTSModel]:
        # NOTE(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == 'tacotron2':
            return Tacotron2
        else:
            raise RuntimeError(
                f'--model must be one of '
                f'{cls.model_choices()}: --frontend {name}')

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> AbsTTSModel:
        if isinstance(args.token_list, str):
            with open(args.token_list) as f:
                token_list = [line.rstrip() for line in f]

            # "args" is saved as it is in a yaml file by BaseTask.main().
            # Overwriting token_list to keep it as "portable".
            args.token_list = token_list.copy()
        elif isinstance(args.token_list, (tuple, list)):
            token_list = args.token_list.copy()
        else:
            raise RuntimeError('token_list must be str or dict')

        vocab_size = len(token_list)
        logging.info(f'Vocabulary size: {vocab_size }')

        model_class = cls.get_model_class(args.lm)
        model = model_class(idim=vocab_size, odim=args.idim, **args.model_conf)
        return model
