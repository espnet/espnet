from pytypes import typechecked

from espnet.nets.pytorch_backend.ctc import CTC
from espnet2.asr.e2e import E2E


class ASRTransformerTask:
    @staticmethod
    @typechecked
    def build_model(
            idim: int, odim: int,
            encoder_kwargs: dict,
            decoder_kwargs: dict,
            ctc_kwargs: dict,
            e2e_kwargs: dict) -> E2E:
        from espnet.nets.pytorch_backend.transformer.decoder import Decoder
        from espnet.nets.pytorch_backend.transformer.encoder import Encoder
        encoder = Encoder(idim=idim, **encoder_kwargs)
        decoder = Decoder(odim=odim, **decoder_kwargs)
        ctc = CTC(odim, **ctc_kwargs, reduce=True)
        model = E2E(
            odim=odim,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            **e2e_kwargs,
        )
        return model


class ASRRNNTask:
    @staticmethod
    @typechecked
    def build_model():
        raise NotImplementedError

