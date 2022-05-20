import numpy as np
import pytest
import torch
import torch_complex.functional as FC
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.beamformer import (
    generalized_eigenvalue_decomposition,
    get_rtf,
    gev_phase_correction,
    signal_framing,
)
from espnet2.enh.layers.complex_utils import solve
from espnet2.layers.stft import Stft

is_torch_1_1_plus = V(torch.__version__) >= V("1.1.0")
is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


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
@pytest.mark.parametrize("mode", ["power", "evd"])
def test_get_rtf(ch, mode):
    if not is_torch_1_9_plus and mode == "evd":
        # torch 1.9.0+ is required for "evd" mode
        return
    if mode == "evd":
        complex_wrapper = torch.complex
        complex_module = torch
    else:
        complex_wrapper = ComplexTensor
        complex_module = FC
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
    ilens = torch.LongTensor([16, 12])
    # (B, T, C, F) -> (B, F, C, T)
    X = complex_wrapper(*torch.unbind(stft(x, ilens)[0], dim=-1)).transpose(-1, -3)
    # (B, F, C, C)
    Phi_X = complex_module.einsum("...ct,...et->...ce", [X, X.conj()])

    is_singular = True
    while is_singular:
        N = complex_wrapper(torch.randn_like(X.real), torch.randn_like(X.imag))
        Phi_N = complex_module.einsum("...ct,...et->...ce", [N, N.conj()])
        is_singular = not np.all(np.linalg.matrix_rank(Phi_N.numpy()) == ch)

    # (B, F, C, 1)
    rtf = get_rtf(Phi_X, Phi_N, mode=mode, reference_vector=0, iterations=20)
    if is_torch_1_1_plus:
        rtf = rtf / (rtf.abs().max(dim=-2, keepdim=True).values + 1e-15)
    else:
        rtf = rtf / (rtf.abs().max(dim=-2, keepdim=True)[0] + 1e-15)
    # rtf \approx Phi_N MaxEigVec(Phi_N^-1 @ Phi_X)
    if is_torch_1_1_plus:
        # torch.solve is required, which is only available after pytorch 1.1.0+
        mat = solve(Phi_X, Phi_N)[0]
        max_eigenvec = solve(rtf, Phi_N)[0]
    else:
        mat = complex_module.matmul(Phi_N.inverse2(), Phi_X)
        max_eigenvec = complex_module.matmul(Phi_N.inverse2(), rtf)
    factor = complex_module.matmul(mat, max_eigenvec)
    assert complex_module.allclose(
        complex_module.matmul(max_eigenvec, factor.transpose(-1, -2)),
        complex_module.matmul(factor, max_eigenvec.transpose(-1, -2)),
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


@pytest.mark.skipif(not is_torch_1_9_plus, reason="Require torch 1.9.0+")
@pytest.mark.parametrize("ch", [2, 4, 6, 8])
def test_gevd(ch):
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
    ilens = torch.LongTensor([16, 12])
    # (B, T, C, F) -> (B, F, C, T)
    X = torch.complex(*torch.unbind(stft(x, ilens)[0], dim=-1)).transpose(-1, -3)
    # (B, F, C, C)
    Phi_X = torch.einsum("...ct,...et->...ce", [X, X.conj()])

    is_singular = True
    while is_singular:
        N = torch.randn_like(X)
        Phi_N = torch.einsum("...ct,...et->...ce", [N, N.conj()])
        is_singular = not torch.linalg.matrix_rank(Phi_N).eq(ch).all()
    # Phi_N = torch.eye(ch, dtype=Phi_X.dtype).view(1, 1, ch, ch).expand_as(Phi_X)

    # e_val: (B, F, C)
    # e_vec: (B, F, C, C)
    e_val, e_vec = generalized_eigenvalue_decomposition(Phi_X, Phi_N)
    e_val = e_val.to(dtype=e_vec.dtype)
    assert torch.allclose(
        torch.matmul(Phi_X, e_vec),
        torch.matmul(torch.matmul(Phi_N, e_vec), e_val.diag_embed()),
    )


@pytest.mark.skipif(not is_torch_1_9_plus, reason="Require torch 1.9.0+")
def test_gev_phase_correction():
    mat = ComplexTensor(torch.rand(2, 3, 4), torch.rand(2, 3, 4))
    mat_th = torch.complex(mat.real, mat.imag)
    norm = gev_phase_correction(mat)
    norm_th = gev_phase_correction(mat_th)
    assert np.allclose(norm.numpy(), norm_th.numpy())
