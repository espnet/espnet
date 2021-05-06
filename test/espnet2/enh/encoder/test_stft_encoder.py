from distutils.version import LooseVersion

import pytest
import torch

from espnet2.enh.encoder.stft_encoder import STFTEncoder

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2.0")


@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("win_length", [512])
@pytest.mark.parametrize("hop_length", [128])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("center", [True])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
def test_STFTEncoder_backward(
    n_fft, win_length, hop_length, window, center, normalized, onesided
):
    if not is_torch_1_2_plus:
        pytest.skip("Pytorch Version Under 1.2 is not supported for Enh task")

    encoder = STFTEncoder(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
    )

    x = torch.rand(2, 32000, requires_grad=True)
    x_lens = torch.tensor([32000, 30000], dtype=torch.long)
    y, flens = encoder(x, x_lens)
    y.abs().sum().backward()
