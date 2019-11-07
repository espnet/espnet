import argparse

import configargparse
from pytypes import typechecked

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet2.asr.e2e import E2E
from espnet2.train.base_task import BaseTask


class ASRRNNTask(BaseTask):
    @classmethod
    @typechecked
    def add_arguments(cls, parser: argparse.ArgumentParser = None) \
            -> argparse.ArgumentParser:
        # Note(kamo): Use '_' instead of '-' to avoid confusion for separator
        if parser is None:
            parser = configargparse.ArgumentParser(description='')
        BaseTask.add_arguments(parser)
        return parser

    @classmethod
    @typechecked
    def build_model(cls, idim: int, odim: int, args: argparse.Namespace):
        # TODO(kamo): Create Encoder and Decoder class to fit the interface.
        raise NotImplementedError

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
