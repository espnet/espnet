import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.partially_AR_model import PartiallyARInference


@pytest.mark.parametrize(
    "interctc_layer_idx, interctc_use_conditioning, interctc_weight",
    [
        ([], False, 0.0),
        ([1], True, 0.5),
    ],
)
def test_partially_ar(interctc_layer_idx, interctc_use_conditioning, interctc_weight):
    vocab_size = 5
    enc_out = 4
    encoder = ConformerEncoder(  # noqa
        20,
        output_size=enc_out,
        linear_units=4,
        num_blocks=2,
        interctc_layer_idx=interctc_layer_idx,
        interctc_use_conditioning=interctc_use_conditioning,
    )
    decoder = TransformerDecoder(
        vocab_size,
        enc_out,
        linear_units=4,
        num_blocks=2,
    )
    ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)

    model = PartiallyARInference(
        ctc=ctc,
        decoder=decoder,
        threshold_probability=0.95,
        sos=4,
        eos=4,
        mask_token=5,
        token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        weights={"decoder": 1.0},
        scorers={"decoder": decoder},
    )

    with torch.no_grad():
        model.eval()

        inputs = dict(
            enc_out=torch.randn(2, 4),
        )
        model(**inputs)
