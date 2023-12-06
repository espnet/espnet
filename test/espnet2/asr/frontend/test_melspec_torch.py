import pytest
import torch

from espnet2.asr.frontend.melspec_torch import MelSpectrogramTorch
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed


def test_frontend_repr():
    frontend = MelSpectrogramTorch()
    print(frontend)


def test_frontend_output_size():
    frontend = MelSpectrogramTorch(n_mels=40)
    assert frontend.output_size() == 40


@pytest.mark.parametrize("normalize", ["mn", None])
@pytest.mark.parametrize("window_fn", ["hamming", "hann"])
@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_frontend_forward(normalize, window_fn, log, train):
    frontend = MelSpectrogramTorch(
        preemp=True,
        n_fft=512,
        log=log,
        win_length=400,
        hop_length=160,
        f_min=20,
        f_max=7600,
        n_mels=20,
        window_fn=window_fn,
        mel_scale="htk",
        normalize=normalize,
    )
    if train:
        frontend.train()
    else:
        frontend.eval()
    set_all_random_seed(14)
    x = torch.randn(2, 1000, requires_grad=True)
    x_lengths = torch.LongTensor([1000, 980])
    y, y_lengths = frontend(x, x_lengths)
