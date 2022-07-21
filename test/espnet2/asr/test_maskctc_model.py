import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.maskctc_model import MaskCTCInference, MaskCTCModel


@pytest.mark.parametrize("encoder_arch", [TransformerEncoder, ConformerEncoder])
@pytest.mark.parametrize(
    "interctc_layer_idx, interctc_use_conditioning, interctc_weight",
    [([], False, 0.0), ([1], True, 0.5),],
)
def test_maskctc(
    encoder_arch, interctc_layer_idx, interctc_use_conditioning, interctc_weight
):
    vocab_size = 5
    enc_out = 4
    encoder = encoder_arch(
        20,
        output_size=enc_out,
        linear_units=4,
        num_blocks=2,
        interctc_layer_idx=interctc_layer_idx,
        interctc_use_conditioning=interctc_use_conditioning,
    )
    decoder = MLMDecoder(vocab_size, enc_out, linear_units=4, num_blocks=2,)
    ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)

    model = MaskCTCModel(
        vocab_size,
        token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
        interctc_weight=interctc_weight,
    )

    inputs = dict(
        speech=torch.randn(2, 10, 20, requires_grad=True),
        speech_lengths=torch.tensor([10, 8], dtype=torch.long),
        text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_lengths=torch.tensor([4, 3], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()

    with torch.no_grad():
        model.eval()

        s2t = MaskCTCInference(
            asr_model=model, n_iterations=2, threshold_probability=0.5,
        )

        # free running
        inputs = dict(enc_out=torch.randn(2, 4),)
        s2t(**inputs)
