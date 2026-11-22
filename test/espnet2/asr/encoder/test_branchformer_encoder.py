import pytest
import torch

from espnet2.asr.encoder.branchformer_encoder import BranchformerEncoder


@pytest.mark.parametrize(
    "input_layer",
    ["linear", "conv2d", "conv2d1", "conv2d2", "conv2d6", "conv2d8", "embed"],
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


def test_branchformer_output_does_not_depend_on_batch_neighbours():
    """Padded frames are masked before the cgMLP convolution."""
    from test.espnet2.asr.encoder.padding_invariance import (
        COMMON,
        D_IN,
        assert_alone_equals_batched,
    )

    torch.manual_seed(0)
    assert_alone_equals_batched(
        BranchformerEncoder(
            D_IN,
            cgmlp_linear_units=64,
            cgmlp_conv_kernel=7,
            **{k: v for k, v in COMMON.items() if k != "linear_units"},
        )
    )


@pytest.mark.parametrize(
    "attn, pos_enc",
    [("selfattn", "abs_pos"), ("rel_selfattn", "rel_pos")],
)
def test_use_sdpa_reaches_the_attention_and_matches_the_default(attn, pos_enc):
    """`use_sdpa` has to be forwarded, and must not change the output.

    The flag reaches PyTorch's scaled_dot_product_attention; checking every
    module that carries it, rather than one attribute name, keeps this honest
    if a layer gains a second attention.
    """
    x = torch.randn(2, 120, 80)
    lengths = torch.tensor([120, 95])

    def build(**kwargs):
        torch.manual_seed(0)
        return BranchformerEncoder(
            80,
            output_size=32,
            attention_heads=2,
            cgmlp_linear_units=32,
            num_blocks=2,
            attention_layer_type=attn,
            pos_enc_layer_type=pos_enc,
            rel_pos_type="latest",
            use_flash_attn=False,
            **kwargs,
        )

    # eval() so dropout does not make the two forward passes draw different
    # masks: the point of the comparison is the attention implementation.
    sdpa_encoder = build(use_sdpa=True).eval()
    attentions = [m for m in sdpa_encoder.modules() if hasattr(m, "use_sdpa")]
    assert attentions and all(m.use_sdpa for m in attentions)

    with torch.no_grad():
        default_out, _, _ = build().eval()(x, lengths)
        sdpa_out, _, _ = sdpa_encoder(x, lengths)
    assert default_out.shape == sdpa_out.shape
    assert torch.allclose(default_out, sdpa_out, atol=1e-5)
