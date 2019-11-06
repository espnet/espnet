#!/usr/bin/env python3
import argparse

import configargparse
from pytypes import typechecked

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet2.asr.e2e import E2E
from espnet2.train.base_task import BaseTask
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder


class ASRTransformerTask(BaseTask):
    @classmethod
    @typechecked
    def get_parser(cls, cmd=None) -> argparse.ArgumentParser:
        # Note(kamo): Use '_' instead of '-' to avoid confusion for separator
        parser = configargparse.ArgumentParser(description='')
        BaseTask.get_parser(parser)
        return parser

    @classmethod
    @typechecked
    def build_model(cls, idim: int, odim: int, args: argparse.Namespace) -> E2E:
        encoder = Encoder(idim=idim,
                          **args.encoder_kwargs)
        decoder = Decoder(odim=odim, **args.decoder_kwargs)
        ctc = CTC(odim, **args.ctc_kwargs, reduce=True)
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
