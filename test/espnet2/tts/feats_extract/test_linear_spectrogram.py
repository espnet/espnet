import numpy as np
import torch

from espnet2.tts.feats_extract.linear_spectrogram import LinearSpectrogram
from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank


def test_forward():
    layer = LinearSpectrogram(n_fft=4, hop_length=1)
    x = torch.randn(2, 4, 9)
    y, _ = layer(x, torch.LongTensor([4, 3]))
    assert y.shape == (2, 5, 9, 3)


def test_backward_leaf_in():
    layer = LinearSpectrogram(n_fft=4, hop_length=1)
    x = torch.randn(2, 4, 9, requires_grad=True)
    y, _ = layer(x, torch.LongTensor([4, 3]))
    y.sum().backward()


def test_backward_not_leaf_in():
    layer = LinearSpectrogram(n_fft=4, hop_length=1)
    x = torch.randn(2, 4, 9, requires_grad=True)
    x = x + 2
    y, _ = layer(x, torch.LongTensor([4, 3]))
    y.sum().backward()


def test_output_size():
    layer = LinearSpectrogram(n_fft=4, hop_length=1)
    print(layer.output_size())


def test_get_parameters():
    layer = LinearSpectrogram(n_fft=4, hop_length=1)
    print(layer.get_parameters())


def test_log_mel_equal():
    layer1 = LinearSpectrogram(n_fft=4, hop_length=1)
    layer2 = LogMelFbank(n_fft=4, hop_length=1, n_mels=2)
    x = torch.randn(2, 4, 9)
    y1, y1_lens = layer1(x, torch.LongTensor([4, 3]))
    y2, _ = layer2(x, torch.LongTensor([4, 3]))
    y1_2, _ = layer2.logmel(y1, y1_lens)
    np.testing.assert_array_equal(
        y2.detach().cpu().numpy(),
        y1_2.detach().cpu().numpy(),
    )
