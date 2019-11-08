import argparse

import configargparse
import yaml
from pytypes import typechecked

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet2.asr.e2e import E2E
from espnet2.train.base_task import BaseTask
from espnet2.utils.get_default_values import get_defaut_values


class ASRTransformerTask(BaseTask):
    @classmethod
    @typechecked
    def add_arguments(cls, parser: configargparse.ArgumentParser = None) \
            -> configargparse.ArgumentParser:
        # Note(kamo): Use '_' instead of '-' to avoid confusion as separator
        if parser is None:
            parser = configargparse.ArgumentParser(
                description='Train ASR Transformer',
                config_file_parser_class=configargparse.YAMLConfigFileParser)

        BaseTask.add_arguments(parser)
        group = parser.add_argument_group(
            description='Transformer config')
        group.add_argument('--encoder_conf', type=yaml.load, default=dict())
        group.add_argument('--decoder_conf', type=yaml.load, default=dict())
        group.add_argument('--ctc_conf', type=yaml.load, default=dict())
        group.add_argument('--ete_conf', type=yaml.load, default=dict())
        return parser

    @classmethod
    @typechecked
    def get_default_config(cls):
        encoder_config = get_defaut_values(Encoder)
        decoder_config = get_defaut_values(Decoder)
        ctc_config = get_defaut_values(CTC)
        e2e_config = get_defaut_values(E2E)
        ctc_config.pop('reduce')

        config = dict(
            encoder_conf=encoder_config,
            decoder_conf=decoder_config,
            ctc_conf=ctc_config,
            e2e_conf=e2e_config,
            )

        config.update(BaseTask.get_default_config())
        return config

    @classmethod
    @typechecked
    def build_model(cls, idim: int, odim: int,
                    args: argparse.Namespace) -> E2E:
        encoder = Encoder(idim=idim, **args.encoder_conf)
        decoder = Decoder(odim=odim, **args.decoder_conf)
        ctc = CTC(odim=odim, **args.ctc_conf, reduce=True)
        model = E2E(
            odim=odim,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            **args.e2e_conf,
        )

        return model


if __name__ == '__main__':
    ASRTransformerTask.main()
