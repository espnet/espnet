import pytest
import torch

from espnet2.layers.stft import Stft


def test_repr():
    print(Stft())


def test_forward():
    layer = Stft()
    x = torch.randn(2, 1000)
    y, _ = layer(x)
    assert y.shape == (2, 257, 8, 2)
    y, ylen = layer(x, torch.tensor([1000, 980], dtype=torch.long))
    assert (ylen == torch.tensor((8, 8), dtype=torch.long)).all()


def test_backward_leaf_in():
    layer = Stft()
    x = torch.randn(2, 1000, requires_grad=True)
    y, _ = layer(x)
    y.sum().backward()


def test_backward_not_leaf_in():
    layer = Stft()
    x = torch.randn(2, 1000, requires_grad=True)
    x = x + 2
    y, _ = layer(x)
    y.sum().backward()


def test_invsere():
    layer = Stft()
    x = torch.randn(2, 1000, requires_grad=True)
    with pytest.raises(NotImplementedError):
        y, _ = layer.inverse(x)
