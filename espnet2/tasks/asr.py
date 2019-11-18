import argparse
import logging
from typing import Any, Dict, Type, Tuple, Optional

import configargparse
import torch
from typeguard import typechecked

from espnet.nets.pytorch_backend.ctc import CTC
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.model import ASRModel
from espnet2.tasks.base_task import BaseTask
from espnet2.utils.get_default_kwargs import get_defaut_kwargs
from espnet2.utils.types import str_or_none, NestedDictAction


class ASRTask(BaseTask):
    @classmethod
    @typechecked
    def add_arguments(cls, parser: configargparse.ArgumentParser = None) \
            -> configargparse.ArgumentParser:
        # Note(kamo): Use '_' instead of '-' to avoid confusion
        if parser is None:
            parser = configargparse.ArgumentParser(
                description='Train ASR',
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

        BaseTask.add_arguments(parser)
        group = parser.add_argument_group(description='Task related')

        # Note(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default('required')
        required += ['token_list']

        group.add_argument('--token_list', type=str_or_none, default=None,
                           help='A text mapping int-id to token')

        group.add_argument(
            '--frontend', type=str_or_none, default='default',
            choices=cls.frontend_choices(), help='Specify frontend class')
        group.add_argument(
            '--frontend_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for frontend class.')

        group.add_argument(
            '--encoder_decoder', type=str, default='transformer',
            choices=cls.encoder_decoder_choices(),
            help='Specify Encoder-Decoder type')
        group.add_argument(
            '--encoder_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for Encoder class.')
        group.add_argument(
            '--decoder_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for Decoder class.')
        group.add_argument(
            '--ctc_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for CTC class.')
        group.add_argument(
            '--model_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for Model class.')

        return parser

    @classmethod
    @typechecked
    def exclude_opts(cls) -> Tuple[str, ...]:
        return ('vocab_size',) + BaseTask.exclude_opts()

    @classmethod
    @typechecked
    def get_default_config(cls) -> Dict[str, Any]:
        # This method is used only for --print_config

        # 0. Parse command line arguments
        parser = ASRTask.add_arguments()
        args, _ = parser.parse_known_args()

        # 1. Get the default values from class.__init__
        frontend_class = cls.get_frontend_class(args.frontend)
        frontend_conf = get_defaut_kwargs(frontend_class)

        encoder_class, decoder_class = \
            cls.get_encoder_decoder_class(args.encoder_decoder)
        encoder_conf = get_defaut_kwargs(encoder_class)
        decoder_conf = get_defaut_kwargs(decoder_class)
        ctc_conf = get_defaut_kwargs(CTC)
        model_conf = get_defaut_kwargs(ASRModel)

        # 2. Create configuration-dict from command-arguments
        config = vars(args)

        # 3. Update the dict using the inherited configuration from BaseTask
        config.update(BaseTask.get_default_config())

        # 4. Overwrite the default config by the command-arguments
        frontend_conf.update(config['frontend_conf'])
        encoder_conf.update(config['encoder_conf'])
        decoder_conf.update(config['decoder_conf'])
        ctc_conf.update(config['ctc_conf'])

        # 5. Reassign them to the configuration
        config.update(
            frontend_conf=frontend_conf,
            encoder_conf=encoder_conf,
            decoder_conf=decoder_conf,
            ctc_conf=ctc_conf,
            model_conf=model_conf)

        # 6. Excludes the options not to be shown
        for k in cls.exclude_opts():
            config.pop(k)

        return config

    @classmethod
    @typechecked
    def frontend_choices(cls) -> Tuple[Optional[str], ...]:
        choices = ('default',)
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        choices += (None,)
        return choices

    @classmethod
    @typechecked
    def get_frontend_class(cls, name: str) -> Type[AbsFrontend]:
        # Note(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == 'default':
            return DefaultFrontend
        else:
            raise RuntimeError(
                f'--frontend must be one of '
                f'{cls.frontend_choices()}: --frontend {name}')

    @classmethod
    @typechecked
    def encoder_decoder_choices(cls) -> Tuple[str, ...]:
        choices = ('Transformer',)
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        return choices

    @classmethod
    @typechecked
    def get_encoder_decoder_class(cls, name: str) \
            -> Tuple[Type[torch.nn.Module], Type[torch.nn.Module]]:
        # Note(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == 'transformer':
            from espnet.nets.pytorch_backend.transformer.decoder import Decoder
            from espnet.nets.pytorch_backend.transformer.encoder import Encoder
            return Encoder, Decoder

        elif name.lower() == 'rnn':
            raise NotImplementedError

        else:
            raise RuntimeError(
                f'--encoder_decoder must be one of '
                f'{cls.encoder_decoder_choices()}: --encoder_decoder {name}')

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ASRModel:
        if isinstance(args.token_list, str):
            with open(args.token_list) as f:
                token_list = [line.rstrip() for line in f]
            # "args" is saved in a yaml file by BaseTask.main().
            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError('token_list must be str or list')
        vocab_size = len(token_list)
        logging.info(f'Vocabulary size: {vocab_size }')

        # 1. frontend
        frontend_class = cls.get_frontend_class(args.frontend)
        if frontend_class is not None:
            frontend = frontend_class(**args.frontend_conf)
        else:
            frontend = None

        # 2. Encoder, Decoder
        encoder_class, decoder_class = \
            cls.get_encoder_decoder_class(args.encoder_decoder)
        encoder = encoder_class(**args.encoder_conf)
        decoder = decoder_class(odim=vocab_size, **args.decoder_conf)

        # 3. CTC
        ctc = CTC(odim=vocab_size, **args.ctc_conf)

        # 4. RNN-T Decoder (Not implemented)
        rnnt_decoder = None

        # 5. Set them to ASRModel
        model = ASRModel(
            vocab_size=vocab_size,
            frontend=frontend,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            rnnt_decoder=rnnt_decoder,
            token_list=token_list,
            **args.model_conf)

        return model
