import argparse
from typing import Any, Dict, Type, Optional, Tuple

import configargparse
import torch
from typeguard import typechecked

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.frontends.feature_transform import \
    FeatureTransform
from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet2.asr.e2e import E2E
from espnet2.tasks.base_task import BaseTask
from espnet2.utils.get_default_values import get_defaut_values
from espnet2.utils.types import yaml_load, str_or_none


class Stft(torch.nn.Module):
    def __init__(self, n_fft: int, fs: int,
                 win_length_ms: int = 25, hop_length_ms: int = 10):
        super().__init__()
        self.n_fft = n_fft

        self.win_length = win_length_ms * fs / 1000
        self.hop_length = hop_length_ms * fs / 1000

    def forward(self, input, ilens) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError
        return torch.stft(
            input, n_fft=self.n, win_length=self.win_length,
            hop_length=self.hop_length)


class ASRTransformerTask(BaseTask):
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
        group = parser.add_argument_group(description='Transformer config')

        # Note(kamo): add_arguments(..., required=True) can't be used
        # to provide --show_config mode. Instead of it, do as
        required = parser.get_default('required')
        required += ['odim', 'fs']

        group.add_argument('--odim', type=int)
        group.add_argument('--n_fft', type=int, default=512)
        group.add_argument('--fs', type=int)

        group.add_argument('--stft', type=str_or_none, default='stft',
                           choices=['sftf'], help='')
        group.add_argument('--stft_conf', type=yaml_load, default=dict(),
                           help='')
        group.add_argument('--frontend', type=str_or_none, default=None,
                           choices=['wpe_mvdr', None], help='')
        group.add_argument('--frontend_conf', type=yaml_load, default=dict(),
                           help='')
        group.add_argument('--feature_transform', type=str_or_none,
                           default='fbank_mvn', choices=['fbank_mvn'], help='')
        group.add_argument('--feature_transform_conf', type=yaml_load,
                           default=dict(), help='')

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

        parser = ASRTransformerTask.add_arguments()
        args, _ = parser.parse_known_args()

        stft_class = cls.get_stft_class(args.stft)
        stft_conf = get_defaut_values(stft_class)

        if args.frontend is not None:
            frontend_class = cls.get_frontend_class(args.frontend)
            frontend_conf = get_defaut_values(frontend_class)
        else:
            frontend_conf = {}

        feature_transform_class = \
            cls.get_frontend_class(args.feature_transform)
        feature_transform_conf = get_defaut_values(feature_transform_class)

        encoder_conf = get_defaut_values(Encoder)
        decoder_conf = get_defaut_values(Decoder)
        ctc_conf = get_defaut_values(CTC)
        e2e_conf = get_defaut_values(E2E)

        config.update(
            stft=args.stft,
            stft_conf=stft_conf,
            frontend=args.frontend,
            frontend_conf=frontend_conf,
            feature_transform=args.feature_transform,
            feature_transform_conf=feature_transform_conf,
            encoder_conf=encoder_conf,
            decoder_conf=decoder_conf,
            ctc_conf=ctc_conf,
            e2e_conf=e2e_conf,
            )

        return config

    @classmethod
    @typechecked
    def get_stft_class(cls, name: str) -> Type[torch.nn.Module]:
        if name.lower() == 'stft':
            return Stft
        else:
            raise RuntimeError(f'Not supported: --stft {name}')

    @classmethod
    @typechecked
    def get_frontend_class(cls, name: str) -> Type[torch.nn.Module]:
        if name.lower() == 'wpe_mvdr':
            return Frontend
        else:
            raise RuntimeError(f'Not supported: --frontend {name}')

    @classmethod
    @typechecked
    def get_feature_transform_class(cls, name: str) -> Type[torch.nn.Module]:
        if name.lower() == 'fbank_mvn':
            return FeatureTransform
        else:
            raise RuntimeError(f'Not supported: --feature_transform {name}')

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> E2E:
        stft_class = cls.get_stft_class(args.stft)
        stft = stft_class(n_fft=args.n_fft, fs=args.fs, **args.stft_conf)

        if args.frontend is not None:
            frontend_class = cls.get_frontend_class(args.frontend)
            frontend = frontend_class(idim=args.n_fft, **args.frontend_conf)
        else:
            frontend = None

        feature_transform_class = cls.get_frontend_class(
            args.feature_transform)
        feature_transform = feature_transform_class(
            fs=args.fs, n_fft=args.n_fft, **args.feature_transform_conf)

        encoder = Encoder(idim=args.n_mels, **args.encoder_conf)
        decoder = Decoder(odim=args.odim, **args.decoder_conf)
        ctc = CTC(odim=args.odim, **args.ctc_conf)
        model = E2E(
            odim=args.odim,
            stft=stft,
            frontend=frontend,
            feature_transform=feature_transform,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            **args.e2e_conf,
        )

        return model


if __name__ == '__main__':
    ASRTransformerTask.main()
