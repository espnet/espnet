from typing import Tuple, Union

import torch
import torch.nn.functional as F
import torch_complex.functional as FC
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import einsum, matmul, reverse

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


""" WPE pytorch version: Ported from https://github.com/fgnt/nara_wpe
Many functions aren't enough tested"""


def signal_framing(
    signal: Union[torch.Tensor, ComplexTensor],
    frame_length: int,
    frame_step: int,
    pad_value=0,
) -> Union[torch.Tensor, ComplexTensor]:
    """Expands signal into frames of frame_length.

    Args:
        signal : (B * F, D, T)
    Returns:
        torch.Tensor: (B * F, D, T, W)
    """
    if isinstance(signal, ComplexTensor):
        real = signal_framing(signal.real, frame_length, frame_step, pad_value)
        imag = signal_framing(signal.imag, frame_length, frame_step, pad_value)
        return ComplexTensor(real, imag)
    elif is_torch_1_9_plus and torch.is_complex(signal):
        real = signal_framing(signal.real, frame_length, frame_step, pad_value)
        imag = signal_framing(signal.imag, frame_length, frame_step, pad_value)
        return torch.complex(real, imag)

    signal = F.pad(signal, (0, frame_length - 1), "constant", pad_value)
    indices = sum(
        [
            list(range(i, i + frame_length))
            for i in range(0, signal.size(-1) - frame_length + 1, frame_step)
        ],
        [],
    )

    signal = signal[..., indices].view(*signal.size()[:-1], -1, frame_length)
    return signal


def get_power(signal, dim=-2) -> torch.Tensor:
    """Calculates power for `signal`

    Args:
        signal : Single frequency signal
            with shape (F, C, T).
        axis: reduce_mean axis
    Returns:
        Power with shape (F, T)

    """
    power = signal.real ** 2 + signal.imag ** 2
    power = power.mean(dim=dim)
    return power


def get_correlations(
    Y: Union[torch.Tensor, ComplexTensor], inverse_power: torch.Tensor, taps, delay
) -> Tuple[Union[torch.Tensor, ComplexTensor], Union[torch.Tensor, ComplexTensor]]:
    """Calculates weighted correlations of a window of length taps

    Args:
        Y : Complex-valued STFT signal with shape (F, C, T)
        inverse_power : Weighting factor with shape (F, T)
        taps (int): Lenghts of correlation window
        delay (int): Delay for the weighting factor

    Returns:
        Correlation matrix of shape (F, taps*C, taps*C)
        Correlation vector of shape (F, taps, C, C)
    """
    assert inverse_power.dim() == 2, inverse_power.dim()
    assert inverse_power.size(0) == Y.size(0), (inverse_power.size(0), Y.size(0))

    F, C, T = Y.size()

    # Y: (F, C, T) -> Psi: (F, C, T, taps)
    Psi = signal_framing(Y, frame_length=taps, frame_step=1)[
        ..., : T - delay - taps + 1, :
    ]
    # Reverse along taps-axis
    Psi = reverse(Psi, dim=-1)
    Psi_conj_norm = Psi.conj() * inverse_power[..., None, delay + taps - 1 :, None]

    # (F, C, T, taps) x (F, C, T, taps) -> (F, taps, C, taps, C)
    correlation_matrix = einsum("fdtk,fetl->fkdle", Psi_conj_norm, Psi)
    # (F, taps, C, taps, C) -> (F, taps * C, taps * C)
    correlation_matrix = correlation_matrix.reshape(F, taps * C, taps * C)

    # (F, C, T, taps) x (F, C, T) -> (F, taps, C, C)
    correlation_vector = einsum(
        "fdtk,fet->fked", Psi_conj_norm, Y[..., delay + taps - 1 :]
    )

    return correlation_matrix, correlation_vector


def get_filter_matrix_conj(
    correlation_matrix: Union[torch.Tensor, ComplexTensor],
    correlation_vector: Union[torch.Tensor, ComplexTensor],
    eps: float = 1e-10,
) -> Union[torch.Tensor, ComplexTensor]:
    """Calculate (conjugate) filter matrix based on correlations for one freq.

    Args:
        correlation_matrix : Correlation matrix (F, taps * C, taps * C)
        correlation_vector : Correlation vector (F, taps, C, C)
        eps:

    Returns:
        filter_matrix_conj (torch.complex/ComplexTensor): (F, taps, C, C)
    """
    F, taps, C, _ = correlation_vector.size()

    # (F, taps, C1, C2) -> (F, C1, taps, C2) -> (F, C1, taps * C2)
    correlation_vector = (
        correlation_vector.permute(0, 2, 1, 3).contiguous().view(F, C, taps * C)
    )

    eye = torch.eye(
        correlation_matrix.size(-1),
        dtype=correlation_matrix.dtype,
        device=correlation_matrix.device,
    )
    shape = (
        tuple(1 for _ in range(correlation_matrix.dim() - 2))
        + correlation_matrix.shape[-2:]
    )
    eye = eye.view(*shape)
    correlation_matrix += eps * eye

    inv_correlation_matrix = correlation_matrix.inverse()
    # (F, C, taps, C) x (F, taps * C, taps * C) -> (F, C, taps * C)
    stacked_filter_conj = matmul(
        correlation_vector, inv_correlation_matrix.transpose(-1, -2)
    )

    # (F, C1, taps * C2) -> (F, C1, taps, C2) -> (F, taps, C2, C1)
    filter_matrix_conj = stacked_filter_conj.view(F, C, taps, C).permute(0, 2, 3, 1)
    return filter_matrix_conj


def perform_filter_operation(
    Y: Union[torch.Tensor, ComplexTensor],
    filter_matrix_conj: Union[torch.Tensor, ComplexTensor],
    taps,
    delay,
) -> Union[torch.Tensor, ComplexTensor]:
    """perform_filter_operation

    Args:
        Y : Complex-valued STFT signal of shape (F, C, T)
        filter Matrix (F, taps, C, C)
    """
    if isinstance(Y, ComplexTensor):
        complex_module = FC
        pad_func = FC.pad
    elif is_torch_1_9_plus and torch.is_complex(Y):
        complex_module = torch
        pad_func = F.pad
    else:
        raise ValueError(
            "Please update your PyTorch version to 1.9+ for complex support."
        )

    T = Y.size(-1)
    # Y_tilde: (taps, F, C, T)
    Y_tilde = complex_module.stack(
        [
            pad_func(Y[:, :, : T - delay - i], (delay + i, 0), mode="constant", value=0)
            for i in range(taps)
        ],
        dim=0,
    )
    reverb_tail = complex_module.einsum("fpde,pfdt->fet", (filter_matrix_conj, Y_tilde))
    return Y - reverb_tail


def wpe_one_iteration(
    Y: Union[torch.Tensor, ComplexTensor],
    power: torch.Tensor,
    taps: int = 10,
    delay: int = 3,
    eps: float = 1e-10,
    inverse_power: bool = True,
) -> Union[torch.Tensor, ComplexTensor]:
    """WPE for one iteration

    Args:
        Y: Complex valued STFT signal with shape (..., C, T)
        power: : (..., T)
        taps: Number of filter taps
        delay: Delay as a guard interval, such that X does not become zero.
        eps:
        inverse_power (bool):
    Returns:
        enhanced: (..., C, T)
    """
    assert Y.size()[:-2] == power.size()[:-1]
    batch_freq_size = Y.size()[:-2]
    Y = Y.view(-1, *Y.size()[-2:])
    power = power.view(-1, power.size()[-1])

    if inverse_power:
        inverse_power = 1 / torch.clamp(power, min=eps)
    else:
        inverse_power = power

    correlation_matrix, correlation_vector = get_correlations(
        Y, inverse_power, taps, delay
    )
    filter_matrix_conj = get_filter_matrix_conj(correlation_matrix, correlation_vector)
    enhanced = perform_filter_operation(Y, filter_matrix_conj, taps, delay)

    enhanced = enhanced.view(*batch_freq_size, *Y.size()[-2:])
    return enhanced


def wpe(
    Y: Union[torch.Tensor, ComplexTensor], taps=10, delay=3, iterations=3
) -> Union[torch.Tensor, ComplexTensor]:
    """WPE

    Args:
        Y: Complex valued STFT signal with shape (F, C, T)
        taps: Number of filter taps
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:

    Returns:
        enhanced: (F, C, T)

    """
    enhanced = Y
    for _ in range(iterations):
        power = get_power(enhanced)
        enhanced = wpe_one_iteration(Y, power, taps=taps, delay=delay)
    return enhanced
