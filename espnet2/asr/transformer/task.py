import argparse
from pathlib import Path

import configargparse
from pytypes import typechecked

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet2.asr.e2e import E2E
from espnet2.train.base_task import BaseTask
from espnet2.utils.misc import get_arguments_from_func


class ASRTransformerTask(BaseTask):
    @classmethod
    @typechecked
    def add_arguments(cls, parser: configargparse.ArgumentParser = None) \
            -> configargparse.ArgumentParser:
        # Note(kamo): Use '_' instead of '-' to avoid confusion for separator
        if parser is None:
            parser = configargparse.ArgumentParser(description='')

        BaseTask.add_arguments(parser)
        return parser

    @classmethod
    @typechecked
    def get_default_config(cls):
        encoder_config = get_arguments_from_func(Encoder)
        decoder_config = get_arguments_from_func(Decoder)
        ctc_config = get_arguments_from_func(CTC)
        e2e_config = get_arguments_from_func(E2E)

        # encoder_config.pop('idim')
        # decoder_config.pop('odim')
        ctc_config.pop('odim')
        ctc_config.pop('reduce')
        # e2e_config.pop('odim')
        # e2e_config.pop('encoder')
        # e2e_config.pop('decoder')
        # e2e_config.pop('ctc')

        config = dict(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            ctc_config=ctc_config,
            e2e_config=e2e_config,
            )

        return config

    @classmethod
    @typechecked
    def build_model(cls, idim: int, odim: int, args: argparse.Namespace) -> E2E:
        encoder = Encoder(idim=idim,
                          **args.encoder_kwargs)
        decoder = Decoder(odim=odim, **args.decoder_kwargs)
        ctc = CTC(odim=odim, **args.ctc_kwargs, reduce=True)
        model = E2E(
            odim=odim,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            **args.e2e_kwargs,
        )

        initialize(model, args.init)

        return model


if __name__ == '__main__':
    ASRTransformerTask.main()
