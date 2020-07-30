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
    xs = torch.randn(2, 256)
    if not use_token_averaged_f0:
        layer(xs, torch.LongTensor([256, 128]))
    else:
        ds = torch.LongTensor([[3, 0, 2], [3, 0, 0]])
        dlens = torch.LongTensor([3, 1])
        ps, _ = layer(
            xs, torch.LongTensor([256, 128]), durations=ds, durations_lengths=dlens
        )
        assert torch.isnan(ps).sum() == 0


def test_output_size():
    layer = Dio(n_fft=4, hop_length=1, f0min=40, f0max=800, fs="16k")
    print(layer.output_size())


def test_get_parameters():
    layer = Dio(n_fft=4, hop_length=1, f0min=40, f0max=800, fs="16k")
    print(layer.get_parameters())
