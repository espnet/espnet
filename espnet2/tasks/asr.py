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
from espnet2.utils.types import str_or_none, int_or_none, NestedDictAction


class Stft(torch.nn.Module):
    @typechecked
    def __init__(self, fs: int,
                 n_fft: int = 512,
                 win_length_ms: int = 25,
                 hop_length_ms: int = 10,
                 center: bool = True,
                 pad_mode: str = 'reflect',
                 normalized: bool = False,
                 onesided: bool = True
                 ):
        super().__init__()
        self.n_fft = n_fft

        self.win_length = int(win_length_ms * fs / 1000)
        self.hop_length = int(hop_length_ms * fs / 1000)
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided

    def forward(self,
                input: torch.Tensor,
                ilens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # output: (Batch, Freq, NFrame, 2=real_imag)
        output = torch.stft(
            input, n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length, center=self.center,
            pad_mode=self.pad_mode, normalized=self.normalized,
            onesided=self.onesided)

        if self.center:
            pad = self.n_fft // 2
            ilens = ilens + 2 * pad
        olens = (ilens - self.win_length) // self.hop_length + 1

        return output, olens


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

        group.add_argument('--odim', type=int_or_none, default=None)
        group.add_argument(
            '--fs', type=humanfriendly.parse_size,
            default=16000, help='The sampling frequency of input wave')

        group.add_argument(
            '--stft', type=str, default='stft',
            choices=cls.stft_choices(), help='Specify stft class')
        group.add_argument(
            '--stft_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for stft class.')
        group.add_argument(
            '--frontend', type=str_or_none,
            choices=cls.frontend_choices(), help='Specify frontend class')
        group.add_argument(
            '--frontend_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for feature-transform class.')
        group.add_argument(
            '--feature_transform', type=str,
            default='fbank_mvn', choices=cls.feature_transform_choices(),
            help='Specify feature-transform class')
        group.add_argument(
            '--feature_transform_conf',
            action=NestedDictAction, default=dict(),
            help='The keyword arguments for feature-transform class.')

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

        # 4. Excluded the specified options
        for k in cls.exclude_opts():
            config.pop(k)

        # 5. Overwrite the default config by the command-arguments
        stft_conf.update(config['stft_conf'])
        frontend_conf.update(config['frontend_conf'])
        feature_transform_conf.update(config['feature_transform_conf'])
        encoder_conf.update(config['encoder_conf'])
        decoder_conf.update(config['decoder_conf'])
        ctc_conf.update(config['ctc_conf'])

        # FIXME(kamo): A little bit dirty way
        feature_transform_conf.pop('fs', None)

        # 6. Reassign them to the configuration
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
            e2e_conf=e2e_conf)

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
            raise RuntimeError(
                f'--stft must be one of {cls.stft_choices()}: --stft {name}')

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
            raise RuntimeError(
                f'--frontend must be one of '
                f'{cls.frontend_choices()}: --frontend {name}')

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
            raise RuntimeError(
                f'--feature_transform must be one of '
                f'{cls.feature_transform_choices()}: '
                f'--feature_transform {name}')

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
        # 1. Stft
        stft_class = cls.get_stft_class(args.stft)
        stft = stft_class(fs=args.fs, **args.stft_conf)

        # 2. [Option] Frontend
        if args.frontend is not None:
            frontend_class = cls.get_frontend_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
        else:
            frontend = None

        # 3. Feature-transform
        feature_transform_class = \
            cls.get_feature_transform_class(args.feature_transform)
        feature_transform = \
            feature_transform_class(fs=args.fs, **args.feature_transform_conf)

        # 4. Encoder, Decoder
        encoder_class, decoder_class = \
            cls.get_encoder_decoder_class(args.encoder_decoder)
        encoder = encoder_class(**args.encoder_conf)
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
            rnnt_decoder=None,
            **args.e2e_conf)

        return model


if __name__ == '__main__':
    ASRTask.main()
