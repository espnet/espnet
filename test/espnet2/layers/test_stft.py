import pytest
import torch

from espnet2.layers.stft import Stft


def test_repr():
    print(Stft())


def test_forward():
    layer = Stft(win_length=4, hop_length=2, n_fft=4)
    x = torch.randn(2, 30)
    y, _ = layer(x)
    assert y.shape == (2, 16, 3, 2)
    y, ylen = layer(x, torch.tensor([30, 15], dtype=torch.long))
    assert (ylen == torch.tensor((16, 8), dtype=torch.long)).all()


def test_backward_leaf_in():
    layer = Stft()
    x = torch.randn(2, 400, requires_grad=True)
    y, _ = layer(x)
    y.sum().backward()


def test_backward_not_leaf_in():
    layer = Stft()
    x = torch.randn(2, 400, requires_grad=True)
    x = x + 2
    y, _ = layer(x)
    y.sum().backward()


def test_inverse():
    layer = Stft()
    x = torch.randn(2, 400, requires_grad=True)
    with pytest.raises(NotImplementedError):
        y, _ = layer.inverse(x)
