import torch

from espnet2.tts.feats_extract.log_spectrogram import LogSpectrogram


def test_forward():
    layer = LogSpectrogram(n_fft=2)
    x = torch.randn(2, 4, 9)
    y, _ = layer(x, torch.LongTensor([4, 3]))
    assert y.shape == (2, 1, 9, 2)


def test_backward_leaf_in():
    layer = LogSpectrogram(n_fft=2)
    x = torch.randn(2, 4, 9, requires_grad=True)
    y, _ = layer(x, torch.LongTensor([4, 3]))
    y.sum().backward()


def test_backward_not_leaf_in():
    layer = LogSpectrogram(n_fft=2)
    x = torch.randn(2, 4, 9, requires_grad=True)
    x = x + 2
    y, _ = layer(x, torch.LongTensor([4, 3]))
    y.sum().backward()


def test_output_size():
    layer = LogSpectrogram(n_fft=2)
    print(layer.output_size())


def test_get_parameters():
    layer = LogSpectrogram(n_fft=2)
    print(layer.get_parameters())
