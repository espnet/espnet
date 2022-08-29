import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder


@pytest.mark.parametrize(
    "input_layer", ["linear", "conv2d", "conv2d2", "conv2d6", "conv2d8", "embed"]
)
@pytest.mark.parametrize("positionwise_layer_type", ["conv1d", "conv1d-linear"])
@pytest.mark.parametrize(
    "rel_pos_type, pos_enc_layer_type, selfattention_layer_type",
    [
        ("legacy", "abs_pos", "selfattn"),
        ("latest", "rel_pos", "rel_selfattn"),
        ("legacy", "rel_pos", "rel_selfattn"),
        ("legacy", "legacy_rel_pos", "legacy_rel_selfattn"),
    ],
)
@pytest.mark.parametrize(
    "interctc_layer_idx, interctc_use_conditioning",
    [([], False), ([1], False), ([1], True),],
)
@pytest.mark.parametrize("stochastic_depth_rate", [0.0, 0.1, [0.1, 0.1]])
def test_encoder_forward_backward(
    input_layer,
    positionwise_layer_type,
    rel_pos_type,
    pos_enc_layer_type,
    selfattention_layer_type,
    interctc_layer_idx,
    interctc_use_conditioning,
    stochastic_depth_rate,
):
    encoder = ConformerEncoder(
        20,
        output_size=2,
        attention_heads=2,
        linear_units=4,
        num_blocks=2,
        input_layer=input_layer,
        macaron_style=False,
        rel_pos_type=rel_pos_type,
        pos_enc_layer_type=pos_enc_layer_type,
        selfattention_layer_type=selfattention_layer_type,
        activation_type="swish",
        use_cnn_module=True,
        cnn_module_kernel=3,
        positionwise_layer_type=positionwise_layer_type,
        interctc_layer_idx=interctc_layer_idx,
        interctc_use_conditioning=interctc_use_conditioning,
        stochastic_depth_rate=stochastic_depth_rate,
    )
    if input_layer == "embed":
        x = torch.randint(0, 10, [2, 32])
    else:
        x = torch.randn(2, 32, 20, requires_grad=True)
    x_lens = torch.LongTensor([32, 28])
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


def test_encoder_invalid_layer_type():
    with pytest.raises(ValueError):
        ConformerEncoder(20, rel_pos_type="dummy")
    with pytest.raises(ValueError):
        ConformerEncoder(20, pos_enc_layer_type="dummy")
    with pytest.raises(ValueError):
        ConformerEncoder(
            20, pos_enc_layer_type="abc_pos", selfattention_layer_type="dummy"
        )


def test_encoder_invalid_rel_pos_combination():
    with pytest.raises(AssertionError):
        ConformerEncoder(
            20,
            rel_pos_type="latest",
            pos_enc_layer_type="legacy_rel_pos",
            selfattention_layer_type="legacy_rel_sselfattn",
        )
    with pytest.raises(AssertionError):
        ConformerEncoder(
            20,
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="legacy_rel_sselfattn",
        )
    with pytest.raises(AssertionError):
        ConformerEncoder(
            20,
            pos_enc_layer_type="legacy_rel_pos",
            selfattention_layer_type="rel_sselfattn",
        )


def test_encoder_invalid_interctc_layer_idx():
    with pytest.raises(AssertionError):
        ConformerEncoder(
            20, num_blocks=2, interctc_layer_idx=[0, 1],
        )
    with pytest.raises(AssertionError):
        ConformerEncoder(
            20, num_blocks=2, interctc_layer_idx=[1, 2],
        )


def test_encoder_output_size():
    encoder = ConformerEncoder(20, output_size=256)
    assert encoder.output_size() == 256


def test_encoder_invalid_type():
    with pytest.raises(ValueError):
        ConformerEncoder(20, input_layer="fff")


def test_encoder_invalid_stochastic_depth_rate():
    with pytest.raises(ValueError):
        ConformerEncoder(
            20, num_blocks=2, stochastic_depth_rate=[0.1],
        )
    with pytest.raises(ValueError):
        ConformerEncoder(
            20, num_blocks=2, stochastic_depth_rate=[0.1, 0.1, 0.1],
        )
