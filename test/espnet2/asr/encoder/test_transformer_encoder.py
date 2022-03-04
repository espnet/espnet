import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder


@pytest.mark.parametrize("input_layer", ["linear", "conv2d", "embed", None])
@pytest.mark.parametrize("positionwise_layer_type", ["conv1d", "conv1d-linear"])
@pytest.mark.parametrize(
    "interctc_layer_idx, interctc_use_conditioning",
    [
        ([], False),
        ([1], False),
        ([1], True),
    ],
)
def test_Encoder_forward_backward(
    input_layer,
    positionwise_layer_type,
    interctc_layer_idx,
    interctc_use_conditioning,
):
    encoder = TransformerEncoder(
        20,
        output_size=40,
        input_layer=input_layer,
        positionwise_layer_type=positionwise_layer_type,
        interctc_layer_idx=interctc_layer_idx,
        interctc_use_conditioning=interctc_use_conditioning,
    )
    if input_layer == "embed":
        x = torch.randint(0, 10, [2, 10])
    else:
        x = torch.randn(2, 10, 20, requires_grad=True)
    x_lens = torch.LongTensor([10, 8])
    if len(interctc_layer_idx) > 0:
        ctc = None
        if interctc_use_conditioning:
            vocab_size = 5
            output_size = encoder.output_size()
            ctc = CTC(odim=vocab_size, encoder_output_size=output_size)
            encoder.conditioning_layer = torch.nn.Linear(vocab_size, output_size)
        y, _, _ = encoder(x, x_lens, ctc=ctc)
        y = y[0]
    else:
        y, _, _ = encoder(x, x_lens)
    y.sum().backward()


def test_encoder_invalid_interctc_layer_idx():
    with pytest.raises(AssertionError):
        TransformerEncoder(
            20,
            num_blocks=2,
            interctc_layer_idx=[0, 1],
        )
    with pytest.raises(AssertionError):
        TransformerEncoder(
            20,
            num_blocks=2,
            interctc_layer_idx=[1, 2],
        )


def test_Encoder_output_size():
    encoder = TransformerEncoder(20, output_size=256)
    assert encoder.output_size() == 256


def test_Encoder_invalid_type():
    with pytest.raises(ValueError):
        TransformerEncoder(20, input_layer="fff")
