import pytest
import torch

from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed


def test_frontend_repr():
    frontend = DefaultFrontend(fs="16k")
    print(frontend)


def test_frontend_output_size():
    frontend = DefaultFrontend(fs="16k", n_mels=40)
    assert frontend.output_size() == 40


def test_frontend_backward():
    frontend = DefaultFrontend(
        fs=160, n_fft=128, win_length=32, hop_length=32, frontend_conf=None
    )
    x = torch.randn(2, 300, requires_grad=True)
    x_lengths = torch.LongTensor([300, 89])
    y, y_lengths = frontend(x, x_lengths)
    y.sum().backward()


@pytest.mark.parametrize("use_wpe", [True, False])
@pytest.mark.parametrize("use_beamformer", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_frontend_backward_multi_channel(train, use_wpe, use_beamformer):
    frontend = DefaultFrontend(
        fs=300,
        n_fft=128,
        win_length=128,
        frontend_conf={"use_wpe": use_wpe, "use_beamformer": use_beamformer},
    )
    if train:
        frontend.train()
    else:
        frontend.eval()
    set_all_random_seed(14)
    x = torch.randn(2, 1000, 2, requires_grad=True)
    x_lengths = torch.LongTensor([1000, 980])
    y, y_lengths = frontend(x, x_lengths)
    y.sum().backward()
