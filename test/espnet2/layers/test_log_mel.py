import numpy as np
import pytest
import torch

from espnet2.layers.log_mel import LogMel


def test_repr():
    print(LogMel())


def test_forward():
    layer = LogMel(n_fft=16, n_mels=2)
    x = torch.randn(2, 4, 9)
    y, _ = layer(x)
    assert y.shape == (2, 4, 2)
    y, ylen = layer(x, torch.tensor([4, 2], dtype=torch.long))
    assert (ylen == torch.tensor((4, 2), dtype=torch.long)).all()


def test_backward_leaf_in():
    layer = LogMel(n_fft=16, n_mels=2)
    x = torch.randn(2, 4, 9, requires_grad=True)
    y, _ = layer(x)
    y.sum().backward()


def test_backward_not_leaf_in():
    layer = LogMel(n_fft=16, n_mels=2)
    x = torch.randn(2, 4, 9, requires_grad=True)
    x = x + 2
    y, _ = layer(x)
    y.sum().backward()


@pytest.mark.parametrize("log_base", [None, 2.0, 10.0, np.e, 3.0, 0.5])
def test_log_base(log_base):
    layer = LogMel(n_fft=16, n_mels=2, log_base=log_base).double()
    x = torch.linspace(1.0, 2.0, 72, dtype=torch.float64).reshape(2, 4, 9)
    lengths = torch.tensor([4, 2])

    actual, actual_lengths = layer(x, lengths)

    mel = np.maximum(x.numpy() @ layer.melmat.numpy(), 1e-10)
    expected = np.log(mel)
    if log_base is not None:
        expected /= np.log(log_base)
    expected[1, 2:] = 0.0
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)
    assert torch.equal(actual_lengths, lengths)
