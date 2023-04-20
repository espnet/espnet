import pytest
import torch
from packaging.version import parse as V

from espnet2.enh.encoder.stft_encoder import STFTEncoder

is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")


@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("win_length", [512])
@pytest.mark.parametrize("hop_length", [128])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("center", [True])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
@pytest.mark.parametrize("use_builtin_complex", [True, False])
def test_STFTEncoder_backward(
    n_fft,
    win_length,
    hop_length,
    window,
    center,
    normalized,
    onesided,
    use_builtin_complex,
):
    encoder = STFTEncoder(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        use_builtin_complex=use_builtin_complex,
    )

    x = torch.rand(2, 32000, requires_grad=True)
    x_lens = torch.tensor([32000, 30000], dtype=torch.long)
    y, flens = encoder(x, x_lens)
    y.abs().sum().backward()


@pytest.mark.skipif(not is_torch_1_12_1_plus, reason="torch.complex32 is used")
@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("win_length", [512])
@pytest.mark.parametrize("hop_length", [128])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("center", [True])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
@pytest.mark.parametrize("use_builtin_complex", [False])
def test_STFTEncoder_float16_dtype(
    n_fft,
    win_length,
    hop_length,
    window,
    center,
    normalized,
    onesided,
    use_builtin_complex,
):
    encoder = STFTEncoder(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        use_builtin_complex=use_builtin_complex,
    )

    x = torch.rand(2, 32000, dtype=torch.float16, requires_grad=True)
    x_lens = torch.tensor([32000, 30000], dtype=torch.long)
    y, flens = encoder(x, x_lens)
    (y.real.pow(2) + y.imag.pow(2)).sum().backward()
