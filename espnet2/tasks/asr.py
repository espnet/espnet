import argparse
from typing import Any, Dict, Type, Tuple, Optional

import configargparse
import humanfriendly
import torch
from typeguard import typechecked

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.frontends.feature_transform import \
    FeatureTransform
from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.asr.e2e import E2E
from espnet2.tasks.base_task import BaseTask
from espnet2.utils.get_default_values import get_defaut_values
from espnet2.utils.types import yaml_load, str_or_none


class Stft(torch.nn.Module):
    @typechecked
    def __init__(self, n_fft: int, fs: int,
                 win_length_ms: int = 25, hop_length_ms: int = 10):
        super().__init__()
        self.n_fft = n_fft

        self.win_length = win_length_ms * fs / 1000
        self.hop_length = hop_length_ms * fs / 1000

    @torch.no_grad()
    def forward(self, input, ilens) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError
        return torch.stft(
            input, n_fft=self.n, win_length=self.win_length,
            hop_length=self.hop_length)


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
        required += ['odim', 'fs']

        group.add_argument('--odim', type=int)
        group.add_argument('--fs', type=humanfriendly.parse_size,
                           help='The sampling frequency of input wave')
        group.add_argument('--n_fft', type=int, default=512,
                           help='The number of FFT')
        group.add_argument('--feat_dim', type=int, default=80,
                           help='')

        group.add_argument(
            '--stft', type=str, default='stft',
            choices=cls.stft_choices(), help='Specify stft class')
        group.add_argument(
            '--stft_conf', type=yaml_load, default=dict(),
            help='The keyword arguments for stft class.')
        group.add_argument(
            '--frontend', type=str_or_none, choices=cls.frontend_choices(),
            help='Specify frontend class')
        group.add_argument(
            '--frontend_conf', type=yaml_load, default=dict(),
            help='The keyword arguments for feature-transform class.')
        group.add_argument('--feature_transform', type=str,
                           default='fbank_mvn', choices=cls.frontend_choices(),
                           help='Specify feature-transform class')
        group.add_argument(
            '--feature_transform_conf', type=yaml_load, default=dict(),
            help='The keyword arguments for feature-transform class.')

        group.add_argument('--encoder_decoder', type=str,
                           default='transformer',
                           choices=cls.encoder_decoder_choices(),
                           help='Specify Encoder-Decoder type')
        group.add_argument('--encoder_conf', type=yaml_load, default=dict(),
                           help='The keyword arguments for Encoder class.')
        group.add_argument('--decoder_conf', type=yaml_load, default=dict(),
                           help='The keyword arguments for Decoder class.')
        group.add_argument('--ctc_conf', type=yaml_load, default=dict(),
                           help='The keyword arguments for CTC class.')
        group.add_argument('--e2e_conf', type=yaml_load, default=dict(),
                           help='The keyword arguments for ETE class.')

        return parser

    @classmethod
    @typechecked
    def get_default_config(cls) -> Dict[str, Any]:
        config = BaseTask.get_default_config()

        parser = ASRTask.add_arguments()
        args, _ = parser.parse_known_args()

        stft_class = cls.get_stft_class(args.stft)
        stft_conf = get_defaut_values(stft_class)

        if args.frontend is not None:
            frontend_class = cls.get_frontend_class(args.frontend)
            frontend_conf = get_defaut_values(frontend_class)
        else:
            frontend_conf = {}

        feature_transform_class = \
            cls.get_feature_transform_class(args.feature_transform)
        feature_transform_conf = get_defaut_values(feature_transform_class)
        # FIXME(kamo): A little bit dirty way
        feature_transform_conf.pop('n_mels', None)

        encoder_class, decoder_class = \
            cls.get_encoder_decoder_class(args.encoder_decoder)
        encoder_conf = get_defaut_values(encoder_class)
        decoder_conf = get_defaut_values(decoder_class)
        ctc_conf = get_defaut_values(CTC)
        e2e_conf = get_defaut_values(E2E)

        config.update(
            stft=args.stft,
            stft_conf=stft_conf,
            frontend=args.frontend,
            frontend_conf=frontend_conf,
            feature_transform=args.feature_transform,
            feature_transform_conf=feature_transform_conf,
            encoder_decoder=args.encoder_decoder,
            encoder_conf=encoder_conf,
            decoder_conf=decoder_conf,
            ctc_conf=ctc_conf,
            e2e_conf=e2e_conf,
            )

        return config

    @classmethod
    @typechecked
    def stft_choices(cls) -> Tuple[str, ...]:
        choices = ('Stft',)
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        return choices

    @classmethod
    @typechecked
    def get_stft_class(cls, name: str) -> Type[torch.nn.Module]:
        if name.lower() == 'stft':
            return Stft
        else:
            raise RuntimeError(f'Not supported: --stft {name}')

    @classmethod
    @typechecked
    def frontend_choices(cls) -> Tuple[Optional[str], ...]:
        choices = ('wpe_mvdr',)
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        choices += (None,)
        return choices

    @classmethod
    @typechecked
    def get_frontend_class(cls, name: str) -> Type[torch.nn.Module]:
        if name.lower() == 'wpe_mvdr':
            return Frontend
        else:
            raise RuntimeError(f'Not supported: --frontend {name}')

    @classmethod
    @typechecked
    def feature_transform_choices(cls) -> Tuple[str, ...]:
        choices = ('Fbank_MVN',)
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        return choices

    @classmethod
    @typechecked
    def get_feature_transform_class(cls, name: str) -> Type[torch.nn.Module]:
        if name.lower() == 'fbank_mvn':
            return FeatureTransform
        else:
            raise RuntimeError(f'Not supported: --feature_transform {name}')

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
            raise RuntimeError(f'Not supported: --encoder_decoder {name}')

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> torch.nn.Module:
        # 1. Stft
        stft_class = cls.get_stft_class(args.stft)
        stft = stft_class(n_fft=args.n_fft, fs=args.fs, **args.stft_conf)

        # 2. [Option] Frontend
        if args.frontend is not None:
            frontend_class = cls.get_frontend_class(args.frontend)
            frontend = frontend_class(idim=args.n_fft, **args.frontend_conf)
        else:
            frontend = None

        # 3. Feature transform
        feature_transform_class = cls.get_feature_transform_class(
            args.feature_transform)
        feature_transform = feature_transform_class(
            fs=args.fs, n_fft=args.n_fft, n_mels=args.feat_dim,
            **args.feature_transform_conf)

        # 4. Encoder, Decoder
        encoder_class, decoder_class = \
            cls.get_encoder_decoder_class(args.encoder_decoder)
        encoder = encoder_class(idim=args.feat_dim, **args.encoder_conf)
        decoder = decoder_class(odim=args.odim, **args.decoder_conf)

        # 5. CTC
        ctc = CTC(odim=args.odim, **args.ctc_conf)

        # 6. Set them to E2E
        model = E2E(
            odim=args.odim,
            stft=stft,
            frontend=frontend,
            feature_transform=feature_transform,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            **args.e2e_conf)

        return model


if __name__ == '__main__':
    ASRTask.main()
