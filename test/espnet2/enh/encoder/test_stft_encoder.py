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


def test_STFTEncoder_reconfig_for_fs():
    encoder = STFTEncoder(
        n_fft=512,
        win_length=512,
        hop_length=256,
        window="hann",
        center=True,
        normalized=False,
        onesided=True,
        use_builtin_complex=True,
        default_fs=16000,
    )

    x = torch.rand(1, 8000, dtype=torch.float32)
    ilens = torch.tensor([8000], dtype=torch.long)
    y_8k, _ = encoder(x, ilens, fs=8000)

    x = torch.rand(1, 32000, dtype=torch.float32)
    y_32k, _ = encoder(x, ilens * 4, fs=32000)

    x = torch.rand(1, 16000, dtype=torch.float32)
    y_16k, flens = encoder(x, ilens * 2)

    assert y_16k.size(0) == y_8k.size(0) == y_32k.size(0)
    assert y_16k.size(1) == y_8k.size(1) == y_32k.size(1)
    assert y_16k.size(-1) == y_8k.size(-1) * 2 - 1
    assert y_32k.size(-1) == y_16k.size(-1) * 2 - 1
