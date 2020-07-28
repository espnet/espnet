import pytest
import torch

from espnet2.tts.feats_extract.dio import Dio


@pytest.mark.parametrize("use_continuous_f0", [False, True])
@pytest.mark.parametrize("use_log_f0", [False, True])
@pytest.mark.parametrize("use_token_averaged_f0", [False, True])
def test_forward(use_continuous_f0, use_log_f0, use_token_averaged_f0):
    layer = Dio(
        n_fft=128,
        hop_length=64,
        f0min=40,
        f0max=800,
        fs="16k",
        use_continuous_f0=use_continuous_f0,
        use_log_f0=use_log_f0,
        use_token_averaged_f0=use_token_averaged_f0,
    )
    x = torch.randn(2, 256)
    if not use_token_averaged_f0:
        layer(x, torch.LongTensor([256, 128]))
    else:
        d = torch.LongTensor([[1, 2, 2], [3, 0, 0]])
        dlens = torch.LongTensor([3, 1])
        layer(x, torch.LongTensor([256, 128]), durations=d, durations_lengths=dlens)


def test_output_size():
    layer = Dio(n_fft=4, hop_length=1, f0min=40, f0max=800, fs="16k")
    print(layer.output_size())


def test_get_parameters():
    layer = Dio(n_fft=4, hop_length=1, f0min=40, f0max=800, fs="16k")
    print(layer.get_parameters())
