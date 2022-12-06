import pytest
import torch

from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder


@pytest.mark.parametrize(
    "input_layer", ["linear", "conv2d", "conv2d2", "conv2d6", "conv2d8", "embed"]
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
@pytest.mark.parametrize("max_pos_emb_len", [128, 5000])
@pytest.mark.parametrize("use_ffn", [True, False])
@pytest.mark.parametrize("macaron_ffn", [True, False])
@pytest.mark.parametrize("linear_units", [1024, 2048])
@pytest.mark.parametrize("merge_conv_kernel", [3, 31])
@pytest.mark.parametrize("layer_drop_rate", [0.0, 0.1])
def test_encoder_forward_backward(
    input_layer,
    use_linear_after_conv,
    rel_pos_type,
    pos_enc_layer_type,
    attention_layer_type,
    max_pos_emb_len,
    use_ffn,
    macaron_ffn,
    linear_units,
    merge_conv_kernel,
    layer_drop_rate,
):
    encoder = EBranchformerEncoder(
        20,
        output_size=2,
        attention_heads=2,
        attention_layer_type=attention_layer_type,
        pos_enc_layer_type=pos_enc_layer_type,
        rel_pos_type=rel_pos_type,
        cgmlp_linear_units=4,
        cgmlp_conv_kernel=3,
        use_linear_after_conv=use_linear_after_conv,
        gate_activation="identity",
        num_blocks=2,
        input_layer=input_layer,
        max_pos_emb_len=max_pos_emb_len,
        use_ffn=use_ffn,
        macaron_ffn=macaron_ffn,
        linear_units=linear_units,
        merge_conv_kernel=merge_conv_kernel,
        layer_drop_rate=layer_drop_rate,
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
        EBranchformerEncoder(20, input_layer="dummy")
    with pytest.raises(ValueError):
        EBranchformerEncoder(20, rel_pos_type="dummy")
    with pytest.raises(ValueError):
        EBranchformerEncoder(20, pos_enc_layer_type="dummy")
    with pytest.raises(ValueError):
        EBranchformerEncoder(
            20, pos_enc_layer_type="abc_pos", attention_layer_type="dummy"
        )
    with pytest.raises(ValueError):
        EBranchformerEncoder(20, positionwise_layer_type="dummy")


def test_encoder_invalid_rel_pos_combination():
    with pytest.raises(AssertionError):
        EBranchformerEncoder(
            20,
            rel_pos_type="latest",
            pos_enc_layer_type="legacy_rel_pos",
            attention_layer_type="legacy_rel_sselfattn",
        )
    with pytest.raises(AssertionError):
        EBranchformerEncoder(
            20,
            pos_enc_layer_type="rel_pos",
            attention_layer_type="legacy_rel_sselfattn",
        )
    with pytest.raises(AssertionError):
        EBranchformerEncoder(
            20,
            pos_enc_layer_type="legacy_rel_pos",
            attention_layer_type="rel_sselfattn",
        )
    with pytest.raises(AssertionError):
        EBranchformerEncoder(
            20,
            attention_layer_type="fast_selfattn",
            pos_enc_layer_type="rel_pos",
        )


def test_encoder_output_size():
    encoder = EBranchformerEncoder(20, output_size=256)
    assert encoder.output_size() == 256
