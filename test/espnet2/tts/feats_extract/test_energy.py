import pytest
import torch

from espnet2.tts.feats_extract.energy import Energy


@pytest.mark.parametrize("use_token_averaged_energy", [False, True])
def test_forward(use_token_averaged_energy):
    layer = Energy(
        n_fft=128,
        hop_length=64,
        fs="16k",
        use_token_averaged_energy=use_token_averaged_energy,
    )
    x = torch.randn(2, 256)
    if not use_token_averaged_energy:
        layer(x, torch.LongTensor([256, 128]))
    else:
        d = torch.LongTensor([[1, 2, 2], [3, 0, 0]])
        dlens = torch.LongTensor([3, 1])
        layer(x, torch.LongTensor([256, 128]), durations=d, durations_lengths=dlens)


def test_output_size():
    layer = Energy(n_fft=4, hop_length=1, fs="16k")
    print(layer.output_size())


def test_get_parameters():
    layer = Energy(n_fft=4, hop_length=1, fs="16k")
    print(layer.get_parameters())
