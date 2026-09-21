"""Shared check: an utterance must encode the same alone and beside a longer one.

Three things used to break this: the subsampling mask sliced from the padded
width (#6633), and the time convolutions of the conformer convolution module
and of cgMLP reading padded frames (#6660). With both, the encoders are
invariant to their batch neighbours up to float rounding, which is what makes
utterance-batched decoding reproduce `--batch_size 1`.
"""

import torch

D_IN, T_SHORT, T_LONG = 40, 120, 163
COMMON = dict(
    output_size=32,
    attention_heads=2,
    linear_units=64,
    num_blocks=2,
    input_layer="conv2d",
)


def padded_inputs():
    """A long and a short utterance, the short one zero-padded to the long."""
    torch.manual_seed(0)
    feats = torch.randn(2, T_LONG, D_IN)
    feats[1, T_SHORT:] = 0.0
    return feats, torch.tensor([T_LONG, T_SHORT])


def alone_vs_batched(encoder):
    """Return (batched, alone, n) for the short utterance, n its real length."""
    encoder.eval()
    feats, lens = padded_inputs()
    with torch.no_grad():
        batched, olens_b = encoder(feats, lens)[:2]
        alone, olens_a = encoder(feats[1:, :T_SHORT], lens[1:])[:2]
    assert int(olens_b[1]) == int(olens_a[0])
    n = int(olens_a[0])
    return batched[1, :n], alone[0, :n], n


def assert_alone_equals_batched(encoder):
    batched, alone, _ = alone_vs_batched(encoder)
    torch.testing.assert_close(batched, alone, rtol=1e-5, atol=1e-5)
