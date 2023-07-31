import pytest
import torch

from espnet2.asr.align_refine_model import AlignRefineModel
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder


@pytest.mark.parametrize("encoder_arch", [TransformerEncoder, ConformerEncoder])
@pytest.mark.parametrize("align_refine_k", [1, 4])
@pytest.mark.parametrize("ctc_weight", [0.0, 0.3, 1.0])
def test_align_refine(encoder_arch, align_refine_k: int, ctc_weight: float):
    vocab_size = 5
    enc_out = 4
    encoder = encoder_arch(
        20,
        output_size=enc_out,
        linear_units=4,
        num_blocks=2,
    )
    decoder = TransformerDecoder(
        vocab_size,
        enc_out,
        linear_units=4,
        num_blocks=2,
        use_output_layer=False,
    )
    ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)

    model = AlignRefineModel(
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
        ctc_weight=ctc_weight,
        align_refine_k=align_refine_k,
    )

    inputs = dict(
        speech=torch.randn(2, 10, 20, requires_grad=True),
        speech_lengths=torch.tensor([10, 8], dtype=torch.long),
        text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_lengths=torch.tensor([4, 3], dtype=torch.long),
    )
    loss, stats, weight = model(**inputs)
    loss.backward()

    # check stats
    if ctc_weight != 0:
        assert "loss_ctc" in stats
    else:
        assert "loss_ctc" not in stats

    if ctc_weight == 1.0:
        assert "loss_align_refine" not in stats
        for i in range(align_refine_k):
            assert f"loss_ar_{i}" not in stats
    else:
        assert "loss_align_refine" in stats
        for i in range(align_refine_k):
            assert f"loss_ar_{i}" in stats
