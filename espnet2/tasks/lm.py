import argparse
import logging
from typing import Any, Dict, Type, Tuple, Optional

import configargparse
from typeguard import typechecked

from espnet2.lm.abs_model import AbsLM
from espnet2.lm.contoller import LanguageModelController
from espnet2.lm.seq_rnn import SequentialRNNLM
from espnet2.tasks.abs_task import AbsTask
from espnet2.utils.get_default_kwargs import get_defaut_kwargs
from espnet2.utils.types import str_or_none
from espnet2.utils.nested_dict_action import NestedDictAction


class LMTask(AbsTask):
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
        required += ['token_list']

        group.add_argument('--token_list', type=str_or_none, default=None,
                           help='A text mapping int-id to token')
        group.add_argument(
            '--lm', type=str, default='seq_rnn',
            choices=cls.lm_choices(), help='Specify lm class')
        group.add_argument(
            '--lm_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for lm class.')
        group.add_argument(
            '--model_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for ModelController class.')

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
        parser = LMTask.add_arguments()
        args, _ = parser.parse_known_args()

        # 1. Get the default values from class.__init__
        lm_class = cls.get_lm_class(args.lm)
        lm_conf = get_defaut_kwargs(lm_class)
        model_conf = get_defaut_kwargs(LanguageModelController)

        # 2. Create configuration-dict from command-arguments
        config = vars(args)

        # 3. Update the dict using the inherited configuration from BaseTask
        config.update(AbsTask.get_default_config())

        # 4. Overwrite the default config by the command-arguments
        lm_conf.update(config['lm_conf'])
        model_conf.update(config['model_conf'])

        # 5. Reassign them to the configuration
        config.update(lm_conf=lm_conf, model_conf=model_conf)

        # 6. Excludes the options not to be shown
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
    def get_lm_class(cls, name: str) -> Type[AbsLM]:
        # NOTE(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == 'seq_rnn':
            return SequentialRNNLM
        else:
            raise RuntimeError(
                f'--lm must be one of '
                f'{cls.lm_choices()}: --lm {name}')

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> LanguageModelController:
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

        # 1. Build LM model
        lm_class = cls.get_lm_class(args.lm)
        lm = lm_class(vocab_size=vocab_size,
                      **args.lm_conf)

        # 2. Build controller
        # Assume the last-id is sos_and_eos
        model = LanguageModelController(lm=lm, sos_and_eos=vocab_size - 1,
                                        **args.model_conf)
        return model
