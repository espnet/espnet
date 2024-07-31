import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.multiconvformer_encoder import MultiConvConformerEncoder


@pytest.mark.parametrize(
    "input_layer",
    ["linear", "conv2d", "conv2d1", "conv2d2", "conv2d6", "conv2d8", "embed"],
)
@pytest.mark.parametrize("use_linear_after_conv", [True, False])
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
    [
        ([], False),
        ([1], False),
        ([1], True),
    ],
)
@pytest.mark.parametrize(
    "multicgmlp_type, multicgmlp_kernel_sizes",
    [
        ("sum", "3"),
        ("sum", "3,5"),
        ("sum", "3,5,7"),
        ("weighted_sum", "3"),
        ("weighted_sum", "3,5"),
        ("weighted_sum", "3,5,7"),
        ("concat", "3"),
        ("concat", "3,5"),
        ("concat", "3,5,7"),
        ("concat_fusion", "3"),
        ("concat_fusion", "3,5"),
        ("concat_fusion", "3,5,7"),
    ],
)
@pytest.mark.parametrize("multicgmlp_merge_conv_kernel", [3, 31])
@pytest.mark.parametrize("stochastic_depth_rate", [0.0, 0.1, [0.1, 0.1]])
def test_encoder_forward_backward(
    input_layer,
    use_linear_after_conv,
    positionwise_layer_type,
    rel_pos_type,
    pos_enc_layer_type,
    selfattention_layer_type,
    interctc_layer_idx,
    interctc_use_conditioning,
    multicgmlp_type,
    multicgmlp_kernel_sizes,
    multicgmlp_merge_conv_kernel,
    stochastic_depth_rate,
):
    encoder = MultiConvConformerEncoder(
        20,
        output_size=2,
        attention_heads=2,
        linear_units=4,
        num_blocks=2,
        input_layer=input_layer,
        selfattention_layer_type=selfattention_layer_type,
        pos_enc_layer_type=pos_enc_layer_type,
        positionwise_layer_type=positionwise_layer_type,
        rel_pos_type=rel_pos_type,
        cgmlp_linear_units=36,
        use_cnn_module=True,
        use_linear_after_conv=use_linear_after_conv,
        gate_activation="identity",
        multicgmlp_type=multicgmlp_type,
        multicgmlp_kernel_sizes=multicgmlp_kernel_sizes,
        multicgmlp_merge_conv_kernel=multicgmlp_merge_conv_kernel,
        interctc_layer_idx=interctc_layer_idx,
        interctc_use_conditioning=interctc_use_conditioning,
        stochastic_depth_rate=stochastic_depth_rate,
    )
    if input_layer == "embed":
        x = torch.randint(0, 10, [2, 32])
    else:
        x = torch.randn(2, 32, 20, requires_grad=True)
    x_lens = torch.LongTensor([32, 28])

    if len(interctc_layer_idx) > 0:  # intermediate CTC
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
        MultiConvConformerEncoder(20, input_layer="dummy")
    with pytest.raises(ValueError):
        MultiConvConformerEncoder(20, rel_pos_type="dummy")
    with pytest.raises(ValueError):
        MultiConvConformerEncoder(20, pos_enc_layer_type="dummy")
    with pytest.raises(ValueError):
        MultiConvConformerEncoder(
            20, pos_enc_layer_type="abc_pos", selfattention_layer_type="dummy"
        )


def test_encoder_invalid_rel_pos_combination():
    with pytest.raises(AssertionError):
        MultiConvConformerEncoder(
            20,
            rel_pos_type="latest",
            pos_enc_layer_type="legacy_rel_pos",
            selfattention_layer_type="legacy_rel_sselfattn",
        )
    with pytest.raises(AssertionError):
        MultiConvConformerEncoder(
            20,
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="legacy_rel_sselfattn",
        )
    with pytest.raises(AssertionError):
        MultiConvConformerEncoder(
            20,
            pos_enc_layer_type="legacy_rel_pos",
            selfattention_layer_type="rel_sselfattn",
        )


def test_encoder_invalid_interctc_layer_idx():
    with pytest.raises(AssertionError):
        MultiConvConformerEncoder(
            20,
            num_blocks=2,
            interctc_layer_idx=[0, 1],
        )
    with pytest.raises(AssertionError):
        MultiConvConformerEncoder(
            20,
            num_blocks=2,
            interctc_layer_idx=[1, 2],
        )


def test_encoder_output_size():
    encoder = MultiConvConformerEncoder(20, output_size=256)
    assert encoder.output_size() == 256


def test_encoder_invalid_type():
    with pytest.raises(ValueError):
        MultiConvConformerEncoder(20, input_layer="fff")


def test_encoder_invalid_stochastic_depth_rate():
    with pytest.raises(ValueError):
        MultiConvConformerEncoder(
            20,
            num_blocks=2,
            stochastic_depth_rate=[0.1],
        )
    with pytest.raises(ValueError):
        MultiConvConformerEncoder(
            20,
            num_blocks=2,
            stochastic_depth_rate=[0.1, 0.1, 0.1],
        )
