"""An utterance must encode the same alone and next to a longer one.

Three things used to break this: the subsampling mask sliced from the padded
width (#6633), and the time convolutions of the conformer convolution module
and of cgMLP reading padded frames (this change). With both, the encoders
below are invariant to their batch neighbours up to float rounding, which is
what makes utterance-batched decoding reproduce `--batch_size 1`.
"""

import pytest
import torch

from espnet2.asr.encoder.branchformer_encoder import BranchformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.asr.encoder.multiconvformer_encoder import MultiConvConformerEncoder
from espnet2.asr.layers.cgmlp import ConvolutionalSpatialGatingUnit
from espnet2.legacy.nets.pytorch_backend.conformer.convolution import (
    ConvolutionModule,
    mask_padded_frames,
)
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask

D_IN, T_SHORT, T_LONG = 40, 120, 163
COMMON = dict(
    output_size=32,
    attention_heads=2,
    linear_units=64,
    num_blocks=2,
    input_layer="conv2d",
)


def _inputs():
    torch.manual_seed(0)
    feats = torch.randn(2, T_LONG, D_IN)
    feats[1, T_SHORT:] = 0.0
    return feats, torch.tensor([T_LONG, T_SHORT])


def _alone_vs_batched(encoder):
    encoder.eval()
    feats, lens = _inputs()
    with torch.no_grad():
        batched, olens_b = encoder(feats, lens)[:2]
        alone, olens_a = encoder(feats[1:, :T_SHORT], lens[1:])[:2]
    assert int(olens_b[1]) == int(olens_a[0])
    n = int(olens_a[0])
    torch.testing.assert_close(batched[1, :n], alone[0, :n], rtol=1e-5, atol=1e-5)


def _encoders():
    yield "conformer", lambda: ConformerEncoder(
        D_IN,
        pos_enc_layer_type="rel_pos",
        selfattention_layer_type="rel_selfattn",
        rel_pos_type="latest",
        cnn_module_kernel=7,
        **COMMON,
    )
    yield "conformer_abs_pos", lambda: ConformerEncoder(
        D_IN,
        pos_enc_layer_type="abs_pos",
        selfattention_layer_type="selfattn",
        cnn_module_kernel=7,
        **COMMON,
    )
    yield "e_branchformer", lambda: EBranchformerEncoder(
        D_IN,
        rel_pos_type="latest",
        cgmlp_linear_units=64,
        cgmlp_conv_kernel=7,
        **COMMON,
    )
    yield "branchformer", lambda: BranchformerEncoder(
        D_IN,
        cgmlp_linear_units=64,
        cgmlp_conv_kernel=7,
        **{k: v for k, v in COMMON.items() if k != "linear_units"},
    )
    for arch in ["sum", "weighted_sum", "concat", "concat_fusion"]:
        yield f"multiconvformer_{arch}", lambda arch=arch: MultiConvConformerEncoder(
            D_IN,
            cgmlp_linear_units=64,
            multicgmlp_type=arch,
            multicgmlp_kernel_sizes="3,7",
            multicgmlp_merge_conv_kernel=7,
            rel_pos_type="latest",
            **COMMON,
        )


@pytest.mark.parametrize(
    "name, build", list(_encoders()), ids=lambda x: x if isinstance(x, str) else ""
)
def test_encoder_output_does_not_depend_on_batch_neighbours(name, build):
    torch.manual_seed(0)
    _alone_vs_batched(build())


def test_legacy_rel_pos_is_documented_as_not_invariant():
    """The legacy relative shift indexes positions by the padded length.

    This is a property of those models, not a bug this change can fix; the
    test pins the fact so that nobody mistakes it for a regression.
    """
    torch.manual_seed(0)
    encoder = ConformerEncoder(
        D_IN,
        pos_enc_layer_type="rel_pos",
        selfattention_layer_type="rel_selfattn",
        rel_pos_type="legacy",
        cnn_module_kernel=7,
        **COMMON,
    ).eval()
    feats, lens = _inputs()
    with torch.no_grad():
        batched, olens_b = encoder(feats, lens)[:2]
        alone, olens_a = encoder(feats[1:, :T_SHORT], lens[1:])[:2]
    n = int(olens_a[0])
    assert int(olens_b[1]) == n
    assert (batched[1, :n] - alone[0, :n]).abs().max() > 1e-3


def test_mask_padded_frames_zeroes_only_the_padding():
    x = torch.randn(2, 3, 5)
    mask = (~make_pad_mask(torch.tensor([5, 3]), maxlen=5)).unsqueeze(1)
    out = mask_padded_frames(x, mask)
    assert torch.equal(out[0], x[0])
    assert torch.equal(out[1, :, :3], x[1, :, :3])
    assert out[1, :, 3:].abs().sum() == 0
    # a mask that cannot apply (another time axis, or None) is ignored
    assert mask_padded_frames(x, None) is x
    assert mask_padded_frames(x, mask[:, :, :1]) is x


def test_convolution_module_masks_before_the_depthwise_conv():
    """Masking the module input would not do: the pointwise conv and the GLU
    write their biases into the padded frames first."""
    torch.manual_seed(0)
    module = ConvolutionModule(8, kernel_size=5).eval()
    x = torch.randn(2, 10, 8)
    x[1, 6:] = 0.0
    mask = (~make_pad_mask(torch.tensor([10, 6]), maxlen=10)).unsqueeze(1)
    with torch.no_grad():
        ref = module(x[1:, :6])
        unmasked = module(x)
        masked = module(x, mask_pad=mask)
    torch.testing.assert_close(masked[1, :6], ref[0], rtol=1e-6, atol=1e-6)
    assert (unmasked[1, :6] - ref[0]).abs().max() > 1e-4
    # without padding the mask changes nothing
    torch.testing.assert_close(masked[0], unmasked[0], rtol=0, atol=0)


def test_csgu_masks_before_the_conv():
    torch.manual_seed(0)
    unit = ConvolutionalSpatialGatingUnit(
        size=8,
        kernel_size=5,
        dropout_rate=0.0,
        use_linear_after_conv=False,
        gate_activation="identity",
    ).eval()
    x = torch.randn(2, 10, 8)
    x[1, 6:] = 0.0
    mask = (~make_pad_mask(torch.tensor([10, 6]), maxlen=10)).unsqueeze(1)
    with torch.no_grad():
        ref = unit(x[1:, :6])
        masked = unit(x, mask_pad=mask)
    torch.testing.assert_close(masked[1, :6], ref[0], rtol=1e-6, atol=1e-6)
