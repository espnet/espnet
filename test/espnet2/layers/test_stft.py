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
    y, _ = layer(x)
    x_lengths = torch.IntTensor([400, 300])
    raw, _ = layer.inverse(y, x_lengths)
    raw, _ = layer.inverse(y)


def test_librosa_stft():
    mkl_is_available = torch.backends.mkl.is_available()
    if not mkl_is_available:
        raise RuntimeError("MKL is not available.")

    layer = Stft()
    layer.eval()
    x = torch.randn(2, 16000, device="cpu")
    y_torch, _ = layer(x)
    torch._C.has_mkl = False
    y_librosa, _ = layer(x)
    assert torch.allclose(y_torch, y_librosa, atol=7e-6)
    torch._C.has_mkl = True
