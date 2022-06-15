import numpy as np
import torch

from espnet2.tts.feats_extract.log_spectrogram import LogSpectrogram
from espnet.transform.spectrogram import spectrogram


def test_forward():
    layer = LogSpectrogram(n_fft=4, hop_length=1)
    x = torch.randn(2, 4, 9)
    y, _ = layer(x, torch.LongTensor([4, 3]))
    assert y.shape == (2, 5, 9, 3)


def test_backward_leaf_in():
    layer = LogSpectrogram(n_fft=4, hop_length=1)
    x = torch.randn(2, 4, 9, requires_grad=True)
    y, _ = layer(x, torch.LongTensor([4, 3]))
    y.sum().backward()


def test_backward_not_leaf_in():
    layer = LogSpectrogram(n_fft=4, hop_length=1)
    x = torch.randn(2, 4, 9, requires_grad=True)
    x = x + 2
    y, _ = layer(x, torch.LongTensor([4, 3]))
    y.sum().backward()


def test_output_size():
    layer = LogSpectrogram(n_fft=4, hop_length=1)
    print(layer.output_size())


def test_get_parameters():
    layer = LogSpectrogram(n_fft=4, hop_length=1)
    print(layer.get_parameters())


def test_compatible_with_espnet1():
    layer = LogSpectrogram(n_fft=16, hop_length=4)
    x = torch.randn(1, 100)
    y, _ = layer(x, torch.LongTensor([100]))
    y = y.numpy()[0]
    y2 = np.log10(spectrogram(x[0].numpy(), n_fft=16, n_shift=4))
    np.testing.assert_allclose(y, y2, rtol=0, atol=1e-4)
