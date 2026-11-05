import pytest
import torch

from espnet2.asr.encoder.tcn_encoder import TCNEncoder

D_IN = 8
SMALL = dict(
    output_size=6, bottleneck_size=4, hidden_size=8, num_blocks=3, num_repeats=2
)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("norm_type", ["cLN", "BN"])
def test_forward_backward(causal, norm_type):
    encoder = TCNEncoder(D_IN, causal=causal, norm_type=norm_type, **SMALL)
    x = torch.randn(2, 20, D_IN, requires_grad=True)
    lens = torch.tensor([20, 13])
    y, olens, states = encoder(x, lens)
    assert y.shape == (2, 20, 6)
    assert torch.equal(olens, lens)
    assert states is None
    assert torch.all(y[1, 13:] == 0)
    y.sum().backward()


def test_output_size():
    assert TCNEncoder(D_IN, **SMALL).output_size() == 6


@pytest.mark.parametrize("norm_type", ["gLN", "GN"])
def test_causal_rejects_time_pooling_norm(norm_type):
    with pytest.raises(ValueError):
        TCNEncoder(D_IN, causal=True, norm_type=norm_type)


@pytest.mark.parametrize("norm_type", ["gLN", "GN"])
def test_non_causal_accepts_time_pooling_norm(norm_type):
    TCNEncoder(D_IN, causal=False, norm_type=norm_type, **SMALL)


@pytest.mark.parametrize("norm_type", ["cLN", "BN"])
def test_causal_output_ignores_future(norm_type):
    """Changing frames from t0 on must leave every output before t0 unchanged."""
    torch.manual_seed(0)
    encoder = TCNEncoder(D_IN, causal=True, norm_type=norm_type, **SMALL).eval()
    t0 = 15
    x = torch.randn(1, 30, D_IN)
    x_future = x.clone()
    x_future[:, t0:] = torch.randn(1, 30 - t0, D_IN)
    lens = torch.tensor([30])
    with torch.no_grad():
        y = encoder(x, lens)[0]
        y_future = encoder(x_future, lens)[0]
    torch.testing.assert_close(y[:, :t0], y_future[:, :t0])
    assert not torch.allclose(y[:, t0:], y_future[:, t0:])


def test_causal_is_batch_invariant():
    """A short utterance encodes the same alone and padded beside a longer one."""
    torch.manual_seed(0)
    encoder = TCNEncoder(D_IN, causal=True, **SMALL).eval()
    x = torch.randn(2, 30, D_IN)
    x[1, 18:] = 0.0
    with torch.no_grad():
        batched = encoder(x, torch.tensor([30, 18]))[0]
        alone = encoder(x[1:, :18], torch.tensor([18]))[0]
    torch.testing.assert_close(batched[1, :18], alone[0])


def test_receptive_field():
    # first block sees P frames, each later block adds (P - 1) * dilation
    encoder = TCNEncoder(D_IN, kernel_size=3, num_blocks=3, num_repeats=2)
    assert encoder.receptive_field == 3 + 2 * (2 + 4) + 2 * (1 + 2 + 4)
