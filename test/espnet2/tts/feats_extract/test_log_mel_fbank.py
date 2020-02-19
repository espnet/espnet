import torch

from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank


def test_forward():
    layer = LogMelFbank(n_fft=2, n_mels=2)
    x = torch.randn(2, 4, 9)
    y, _ = layer(x, torch.LongTensor([4, 3]))
    assert y.shape == (2, 1, 9, 2)


def test_backward_leaf_in():
    layer = LogMelFbank(n_fft=2, n_mels=2)
    x = torch.randn(2, 4, 9, requires_grad=True)
    y, _ = layer(x, torch.LongTensor([4, 3]))
    y.sum().backward()


def test_backward_not_leaf_in():
    layer = LogMelFbank(n_fft=2, n_mels=2)
    x = torch.randn(2, 4, 9, requires_grad=True)
    x = x + 2
    y, _ = layer(x, torch.LongTensor([4, 3]))
    y.sum().backward()


def test_output_size():
    layer = LogMelFbank(n_fft=2, n_mels=2, fs="16k")
    print(layer.output_size())


def test_get_parameters():
    layer = LogMelFbank(n_fft=2, n_mels=2, fs="16k")
    print(layer.get_parameters())
