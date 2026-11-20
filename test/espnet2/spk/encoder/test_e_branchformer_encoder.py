import pytest
import torch

from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.spk.encoder.e_branchformer_encoder import MfaEBranchformerEncoder
from espnet2.tasks.lid import LIDTask


def _encoder(**kwargs):
    torch.manual_seed(0)
    conf = dict(
        input_size=20,
        output_size=32,
        attention_heads=4,
        linear_units=64,
        cgmlp_linear_units=64,
        num_blocks=3,
        input_layer="conv2d",
        attention_layer_type="selfattn",
        pos_enc_layer_type="abs_pos",
        use_ffn=True,
        macaron_ffn=True,
        merge_conv_kernel=3,
        cgmlp_conv_kernel=3,
        use_flash_attn=False,
    )
    conf.update(kwargs)
    return MfaEBranchformerEncoder(**conf).eval()


def test_output_is_channel_first_with_subsampled_lengths():
    encoder = _encoder()
    assert encoder.returns_lengths
    assert encoder.output_size() == 32 * 3
    out, olens = encoder(torch.randn(2, 41, 20), torch.tensor([41, 25]))
    # conv2d subsampling: ((T - 1) // 2 - 1) // 2
    assert out.shape == (2, 96, 9)
    assert olens.tolist() == [9, 5]


def test_no_lengths_means_unpadded():
    encoder = _encoder()
    x = torch.randn(2, 41, 20)
    out, olens = encoder(x)
    ref, _ = encoder(x, torch.tensor([41, 41]))
    assert olens.tolist() == [9, 9]
    torch.testing.assert_close(out, ref)


def test_padding_does_not_change_the_valid_frames():
    encoder = _encoder()
    short = torch.randn(1, 64, 20)
    alone, alone_lens = encoder(short, torch.tensor([64]))
    padded = torch.cat([short, torch.randn(1, 56, 20) * 5], dim=1)
    batch, batch_lens = encoder(padded, torch.tensor([64]))
    assert alone_lens.tolist() == batch_lens.tolist() == [15]
    torch.testing.assert_close(alone, batch[..., :15], atol=1e-5, rtol=1e-5)


def test_output_is_the_normalized_concatenation_of_every_block():
    encoder = _encoder()
    x, lengths = torch.randn(2, 120, 20), torch.tensor([120, 64])
    out, _ = encoder(x, lengths)
    (_, hidden), _, _ = EBranchformerEncoder.forward(
        encoder, x, lengths, return_all_hs=True
    )
    assert len(hidden) == 3
    # The last block is what EBranchformerEncoder normalizes into its output.
    last, _, _ = EBranchformerEncoder.forward(encoder, x, lengths)
    torch.testing.assert_close(encoder.after_norm(hidden[-1]), last)
    torch.testing.assert_close(
        out, encoder.mfa_norm(torch.cat(hidden, dim=-1)).transpose(1, 2)
    )


def test_layer_drop_is_rejected():
    with pytest.raises(ValueError, match="layer_drop_rate"):
        _encoder(layer_drop_rate=0.1)


def test_registered_in_lid_task():
    (encoders,) = [c for c in LIDTask.class_choices_list if c.name == "encoder"]
    assert encoders.get_class("mfa_e_branchformer") is MfaEBranchformerEncoder
