import numpy as np
import torch

from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet.transform.spectrogram import logmelspectrogram


def test_forward():
    layer = LogMelFbank(n_fft=4, hop_length=1, n_mels=2)
    x = torch.randn(2, 4, 9)
    y, _ = layer(x, torch.LongTensor([4, 3]))
    assert y.shape == (2, 5, 9, 2)


def test_backward_leaf_in():
    layer = LogMelFbank(n_fft=4, hop_length=1, n_mels=2)
    x = torch.randn(2, 4, 9, requires_grad=True)
    y, _ = layer(x, torch.LongTensor([4, 3]))
    y.sum().backward()


def test_backward_not_leaf_in():
    layer = LogMelFbank(n_fft=4, hop_length=1, n_mels=2)
    x = torch.randn(2, 4, 9, requires_grad=True)
    x = x + 2
    y, _ = layer(x, torch.LongTensor([4, 3]))
    y.sum().backward()


def test_output_size():
    layer = LogMelFbank(n_fft=4, hop_length=1, n_mels=2, fs="16k")
    print(layer.output_size())


def test_get_parameters():
    layer = LogMelFbank(n_fft=4, hop_length=1, n_mels=2, fs="16k")
    print(layer.get_parameters())


def test_compatible_with_espnet1():
    layer = LogMelFbank(n_fft=16, hop_length=4, n_mels=4, fs="16k", fmin=80, fmax=7600)
    x = torch.randn(1, 100)
    y, _ = layer(x, torch.LongTensor([100]))
    y = y.numpy()[0]
    y2 = logmelspectrogram(
        x[0].numpy(), n_fft=16, n_shift=4, n_mels=4, fs=16000, fmin=80, fmax=7600
    )
    np.testing.assert_allclose(y, y2, rtol=0, atol=1e-5)
