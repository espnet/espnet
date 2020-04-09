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
