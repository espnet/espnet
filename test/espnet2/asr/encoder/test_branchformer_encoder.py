import pytest
import torch

from espnet2.asr.encoder.branchformer_encoder import BranchformerEncoder


@pytest.mark.parametrize(
    "input_layer", ["linear", "conv2d", "conv2d1", "conv2d2", "conv2d6", "conv2d8", "embed"]
)
@pytest.mark.parametrize("use_linear_after_conv", [True, False])
@pytest.mark.parametrize(
    "rel_pos_type, pos_enc_layer_type, attention_layer_type",
    [
        ("legacy", "abs_pos", "selfattn"),
        ("latest", "rel_pos", "rel_selfattn"),
        ("legacy", "rel_pos", "rel_selfattn"),
        ("legacy", "legacy_rel_pos", "legacy_rel_selfattn"),
        ("legacy", "abs_pos", "fast_selfattn"),
    ],
)
@pytest.mark.parametrize(
    "merge_method, cgmlp_weight, attn_branch_drop_rate",
    [
        ("concat", 0.5, 0.0),
        ("learned_ave", 0.5, 0.0),
        ("learned_ave", 0.5, 0.1),
        ("learned_ave", 0.5, [0.1, 0.1]),
        ("fixed_ave", 0.5, 0.0),
        ("fixed_ave", [0.5, 0.5], 0.0),
        ("fixed_ave", 0.0, 0.0),
        ("fixed_ave", 1.0, 0.0),
    ],
)
@pytest.mark.parametrize("stochastic_depth_rate", [0.0, 0.1, [0.1, 0.1]])
def test_encoder_forward_backward(
    input_layer,
    use_linear_after_conv,
    rel_pos_type,
    pos_enc_layer_type,
    attention_layer_type,
    merge_method,
    cgmlp_weight,
    attn_branch_drop_rate,
    stochastic_depth_rate,
):
    encoder = BranchformerEncoder(
        20,
        output_size=2,
        use_attn=True,
        attention_heads=2,
        attention_layer_type=attention_layer_type,
        pos_enc_layer_type=pos_enc_layer_type,
        rel_pos_type=rel_pos_type,
        use_cgmlp=True,
        cgmlp_linear_units=4,
        cgmlp_conv_kernel=3,
        use_linear_after_conv=use_linear_after_conv,
        gate_activation="identity",
        merge_method=merge_method,
        cgmlp_weight=cgmlp_weight,
        attn_branch_drop_rate=attn_branch_drop_rate,
        num_blocks=2,
        input_layer=input_layer,
        stochastic_depth_rate=stochastic_depth_rate,
    )
    if input_layer == "embed":
        x = torch.randint(0, 10, [2, 32])
    else:
        x = torch.randn(2, 32, 20, requires_grad=True)
    x_lens = torch.LongTensor([32, 28])
    y, _, _ = encoder(x, x_lens)
    y.sum().backward()


def test_encoder_invalid_layer_type():
    with pytest.raises(ValueError):
        BranchformerEncoder(20, rel_pos_type="dummy")
    with pytest.raises(ValueError):
        BranchformerEncoder(20, pos_enc_layer_type="dummy")
    with pytest.raises(ValueError):
        BranchformerEncoder(
            20, pos_enc_layer_type="abc_pos", attention_layer_type="dummy"
        )


def test_encoder_invalid_rel_pos_combination():
    with pytest.raises(AssertionError):
        BranchformerEncoder(
            20,
            rel_pos_type="latest",
            pos_enc_layer_type="legacy_rel_pos",
            attention_layer_type="legacy_rel_sselfattn",
        )
    with pytest.raises(AssertionError):
        BranchformerEncoder(
            20,
            pos_enc_layer_type="rel_pos",
            attention_layer_type="legacy_rel_sselfattn",
        )
    with pytest.raises(AssertionError):
        BranchformerEncoder(
            20,
            pos_enc_layer_type="legacy_rel_pos",
            attention_layer_type="rel_sselfattn",
        )
    with pytest.raises(AssertionError):
        BranchformerEncoder(
            20,
            attention_layer_type="fast_selfattn",
            pos_enc_layer_type="rel_pos",
        )


def test_encoder_output_size():
    encoder = BranchformerEncoder(20, output_size=256)
    assert encoder.output_size() == 256


def test_encoder_invalid_type():
    with pytest.raises(ValueError):
        BranchformerEncoder(20, input_layer="fff")


def test_encoder_invalid_cgmlp_weight():
    with pytest.raises(AssertionError):
        BranchformerEncoder(
            20,
            merge_method="fixed_ave",
            cgmlp_weight=-1.0,
        )
    with pytest.raises(ValueError):
        BranchformerEncoder(
            20,
            num_blocks=2,
            cgmlp_weight=[0.1, 0.1, 0.1],
        )


def test_encoder_invalid_merge_method():
    with pytest.raises(ValueError):
        BranchformerEncoder(
            20,
            merge_method="dummy",
        )


def test_encoder_invalid_two_branches():
    with pytest.raises(AssertionError):
        BranchformerEncoder(
            20,
            use_attn=False,
            use_cgmlp=False,
        )


def test_encoder_invalid_attn_branch_drop_rate():
    with pytest.raises(ValueError):
        BranchformerEncoder(
            20,
            num_blocks=2,
            attn_branch_drop_rate=[0.1, 0.1, 0.1],
        )


def test_encoder_invalid_stochastic_depth_rate():
    with pytest.raises(ValueError):
        BranchformerEncoder(
            20,
            num_blocks=2,
            stochastic_depth_rate=[0.1],
        )
    with pytest.raises(ValueError):
        BranchformerEncoder(
            20,
            num_blocks=2,
            stochastic_depth_rate=[0.1, 0.1, 0.1],
        )
