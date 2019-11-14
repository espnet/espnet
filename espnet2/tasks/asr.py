import argparse
from typing import Any, Dict, Type, Tuple, Optional

import configargparse
import torch
from typeguard import typechecked
import humanfriendly

from espnet.nets.pytorch_backend.ctc import CTC
from espnet2.asr.e2e import E2E
from espnet2.asr.frontend import Frontend1
from espnet2.tasks.base_task import BaseTask
from espnet2.utils.get_default_values import get_defaut_values
from espnet2.utils.types import str_or_none, int_or_none, NestedDictAction


class ASRTask(BaseTask):
    @classmethod
    @typechecked
    def add_arguments(cls, parser: configargparse.ArgumentParser = None) \
            -> configargparse.ArgumentParser:
        # Note(kamo): Use '_' instead of '-' to avoid confusion as separator
        if parser is None:
            parser = configargparse.ArgumentParser(
                description='Train ASR Transformer',
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

        BaseTask.add_arguments(parser)
        group = parser.add_argument_group(description='Task related')

        # Note(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default('required')
        required += ['odim']

        group.add_argument(
            '--fs', type=humanfriendly.parse_size, default=16000)
        group.add_argument('--odim', type=int_or_none, default=None)

        group.add_argument(
            '--frontend', type=str_or_none, default='frontend1',
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
            '--e2e_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for ETE class.')

        return parser

    @classmethod
    @typechecked
    def exclude_opts(cls) -> Tuple[str, ...]:
        return ('odim',) + BaseTask.exclude_opts()

    @classmethod
    @typechecked
    def get_default_config(cls) -> Dict[str, Any]:
        # 0. Parse command line arguments
        parser = ASRTask.add_arguments()
        args, _ = parser.parse_known_args()

        # 1. Get the default values from class.__init__
        frontend_class = cls.get_frontend_class(args.frontend)
        frontend_conf = get_defaut_values(frontend_class)

        encoder_class, decoder_class = \
            cls.get_encoder_decoder_class(args.encoder_decoder)
        encoder_conf = get_defaut_values(encoder_class)
        decoder_conf = get_defaut_values(decoder_class)
        ctc_conf = get_defaut_values(CTC)
        e2e_conf = get_defaut_values(E2E)

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
            e2e_conf=e2e_conf)

        # 6. Excludes the specified options
        for k in cls.exclude_opts():
            config.pop(k)

        return config

    @classmethod
    @typechecked
    def frontend_choices(cls) -> Tuple[Optional[str], ...]:
        choices = ('frontend1',)
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        return choices

    @classmethod
    @typechecked
    def get_frontend_class(cls, name: str) -> Type[torch.nn.Module]:
        if name.lower() == 'frontend1':
            return Frontend1
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
    def build_model(cls, args: argparse.Namespace) -> torch.nn.Module:
        # 1. Frontend
        frontend_class = cls.get_frontend_class(args.frontend)
        frontend = frontend_class(**args.frontend_conf)

        # 2. Encoder, Decoder
        encoder_class, decoder_class = \
            cls.get_encoder_decoder_class(args.encoder_decoder)
        encoder = encoder_class(**args.encoder_conf)
        decoder = decoder_class(odim=args.odim, **args.decoder_conf)

        # 3. CTC
        ctc = CTC(odim=args.odim, **args.ctc_conf)

        # 4. Set them to E2E
        model = E2E(
            odim=args.odim,
            frontend=frontend,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            rnnt_decoder=None,
            **args.e2e_conf)

        return model


if __name__ == '__main__':
    # These two are equivalent:
    #   % python -m espnet2.tasks.asr
    #   % python -m espnet2.bin.train asr

    import sys
    from espnet.utils.cli_utils import get_commandline_args
    print(get_commandline_args(), file=sys.stderr)
    ASRTask.main()
