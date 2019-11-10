import argparse
import sys
from typing import Any, Dict

import configargparse
from typeguard import typechecked

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet2.asr.e2e import E2E
from espnet2.train.base_task import BaseTask
from espnet2.utils.get_default_values import get_defaut_values
from espnet2.utils.types import yaml_load


class ASRTransformerTask(BaseTask):
    @classmethod
    @typechecked
    def add_arguments(cls, parser: configargparse.ArgumentParser = None) \
            -> configargparse.ArgumentParser:
        # Note(kamo): Use '_' instead of '-' to avoid confusion as separator
        #   >>> cls.check_required(args, 'output_dir')
        if parser is None:
            parser = configargparse.ArgumentParser(
                description='Train ASR Transformer',
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

        BaseTask.add_arguments(parser)
        group = parser.add_argument_group(
            description='Transformer config')

        # Note(kamo): add_arguments(..., required=True) can't be used
        # to provide --show_config mode. Instead of it, do as
        required = parser.get_default('required')
        required += ['idim', 'odim']

        group.add_argument('--idim', type=int)
        group.add_argument('--odim', type=int)
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
        encoder_conf = get_defaut_values(Encoder)
        decoder_conf = get_defaut_values(Decoder)
        ctc_conf = get_defaut_values(CTC)
        e2e_conf = get_defaut_values(E2E)
        ctc_conf.pop('reduce')

        config.update(
            encoder_conf=encoder_conf,
            decoder_conf=decoder_conf,
            ctc_conf=ctc_conf,
            e2e_conf=e2e_conf,
            )

        return config

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> E2E:
        cls.check_required(args, 'idim')
        cls.check_required(args, 'odim')
        encoder = Encoder(idim=args.idim, **args.encoder_conf)
        decoder = Decoder(odim=args.odim, **args.decoder_conf)
        ctc = CTC(odim=args.odim, **args.ctc_conf, reduce=True)
        model = E2E(
            odim=args.odim,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            **args.e2e_conf,
        )

        return model


if __name__ == '__main__':
    ASRTransformerTask.main()
