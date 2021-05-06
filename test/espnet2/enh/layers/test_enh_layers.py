from distutils.version import LooseVersion

import pytest
import torch
import torch_complex.functional as FC
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.beamformer import get_rtf
from espnet2.enh.layers.beamformer import signal_framing
from espnet2.layers.stft import Stft

is_torch_1_1_plus = LooseVersion(torch.__version__) >= LooseVersion("1.1.0")


random_speech = torch.tensor(
    [
        [
            [0.026, 0.031, 0.023, 0.029, 0.026, 0.029, 0.028, 0.027],
            [0.027, 0.031, 0.023, 0.027, 0.026, 0.028, 0.027, 0.027],
            [0.026, 0.030, 0.023, 0.026, 0.025, 0.028, 0.028, 0.028],
            [0.024, 0.028, 0.024, 0.027, 0.024, 0.027, 0.030, 0.030],
            [0.025, 0.027, 0.025, 0.028, 0.023, 0.026, 0.031, 0.031],
            [0.027, 0.026, 0.025, 0.029, 0.022, 0.026, 0.032, 0.031],
            [0.028, 0.026, 0.024, 0.031, 0.023, 0.025, 0.031, 0.029],
            [0.029, 0.024, 0.023, 0.032, 0.023, 0.024, 0.030, 0.027],
            [0.028, 0.024, 0.023, 0.030, 0.023, 0.023, 0.028, 0.027],
            [0.029, 0.026, 0.023, 0.029, 0.025, 0.024, 0.027, 0.025],
            [0.029, 0.027, 0.024, 0.026, 0.025, 0.027, 0.025, 0.025],
            [0.029, 0.031, 0.026, 0.024, 0.028, 0.028, 0.024, 0.025],
            [0.030, 0.038, 0.029, 0.023, 0.035, 0.032, 0.024, 0.026],
            [0.029, 0.040, 0.030, 0.023, 0.039, 0.039, 0.025, 0.027],
            [0.028, 0.040, 0.032, 0.025, 0.041, 0.039, 0.026, 0.028],
            [0.028, 0.041, 0.039, 0.027, 0.044, 0.041, 0.029, 0.035],
        ],
        [
            [0.015, 0.021, 0.012, 0.006, 0.028, 0.021, 0.024, 0.018],
            [0.005, 0.034, 0.036, 0.017, 0.016, 0.037, 0.011, 0.029],
            [0.011, 0.029, 0.060, 0.029, 0.045, 0.035, 0.034, 0.018],
            [0.031, 0.036, 0.040, 0.037, 0.059, 0.032, 0.035, 0.029],
            [0.031, 0.031, 0.036, 0.029, 0.058, 0.035, 0.039, 0.045],
            [0.050, 0.038, 0.052, 0.052, 0.059, 0.044, 0.055, 0.045],
            [0.025, 0.054, 0.054, 0.047, 0.043, 0.059, 0.045, 0.060],
            [0.042, 0.056, 0.073, 0.029, 0.048, 0.063, 0.051, 0.049],
            [0.053, 0.048, 0.045, 0.052, 0.039, 0.045, 0.031, 0.053],
            [0.054, 0.044, 0.053, 0.031, 0.062, 0.050, 0.048, 0.046],
            [0.053, 0.036, 0.075, 0.046, 0.073, 0.052, 0.045, 0.030],
            [0.039, 0.025, 0.061, 0.046, 0.064, 0.032, 0.027, 0.033],
            [0.053, 0.032, 0.052, 0.033, 0.052, 0.029, 0.026, 0.017],
            [0.054, 0.034, 0.054, 0.033, 0.045, 0.043, 0.024, 0.018],
            [0.031, 0.025, 0.043, 0.016, 0.051, 0.040, 0.023, 0.030],
            [0.008, 0.023, 0.024, 0.019, 0.032, 0.024, 0.012, 0.027],
        ],
    ],
    dtype=torch.double,
)


@pytest.mark.parametrize("ch", [2, 4, 6, 8])
def test_get_rtf(ch):
    stft = Stft(
        n_fft=8,
        win_length=None,
        hop_length=2,
        center=True,
        window="hann",
        normalized=False,
        onesided=True,
    )
    torch.random.manual_seed(0)
    x = random_speech[..., :ch]
    n = torch.rand(2, 16, ch, dtype=torch.double)
    ilens = torch.LongTensor([16, 12])
    # (B, T, C, F) -> (B, F, C, T)
    X = ComplexTensor(*torch.unbind(stft(x, ilens)[0], dim=-1)).transpose(-1, -3)
    N = ComplexTensor(*torch.unbind(stft(n, ilens)[0], dim=-1)).transpose(-1, -3)
    # (B, F, C, C)
    Phi_X = FC.einsum("...ct,...et->...ce", [X, X.conj()])
    Phi_N = FC.einsum("...ct,...et->...ce", [N, N.conj()])
    # (B, F, C, 1)
    rtf = get_rtf(Phi_X, Phi_N, reference_vector=0, iterations=20)
    if is_torch_1_1_plus:
        rtf = rtf / (rtf.abs().max(dim=-2, keepdim=True).values + 1e-15)
    else:
        rtf = rtf / (rtf.abs().max(dim=-2, keepdim=True)[0] + 1e-15)
    # rtf \approx Phi_N MaxEigVec(Phi_N^-1 @ Phi_X)
    if is_torch_1_1_plus:
        # torch.solve is required, which is only available after pytorch 1.1.0+
        mat = FC.solve(Phi_X, Phi_N)[0]
        max_eigenvec = FC.solve(rtf, Phi_N)[0]
    else:
        mat = FC.matmul(Phi_N.inverse2(), Phi_X)
        max_eigenvec = FC.matmul(Phi_N.inverse2(), rtf)
    factor = FC.matmul(mat, max_eigenvec)
    assert FC.allclose(
        FC.matmul(max_eigenvec, factor.transpose(-1, -2)),
        FC.matmul(factor, max_eigenvec.transpose(-1, -2)),
    )


def test_signal_framing():
    # tap length = 1
    taps, delay = 0, 1
    X = ComplexTensor(torch.rand(2, 10, 6, 20), torch.rand(2, 10, 6, 20))
    X2 = signal_framing(X, taps + 1, 1, delay, do_padding=False)
    assert FC.allclose(X, X2.squeeze(-1))

    # tap length > 1, no padding
    taps, delay = 5, 3
    X = ComplexTensor(torch.rand(2, 10, 6, 20), torch.rand(2, 10, 6, 20))
    X2 = signal_framing(X, taps + 1, 1, delay, do_padding=False)
    assert X2.shape == torch.Size([2, 10, 6, 20 - taps - delay + 1, taps + 1])
    assert FC.allclose(X2[..., 0], X[..., : 20 - taps - delay + 1])

    # tap length > 1, padding
    taps, delay = 5, 3
    X = ComplexTensor(torch.rand(2, 10, 6, 20), torch.rand(2, 10, 6, 20))
    X2 = signal_framing(X, taps + 1, 1, delay, do_padding=True)
    assert X2.shape == torch.Size([2, 10, 6, 20, taps + 1])
    assert FC.allclose(X2[..., -1], X)
