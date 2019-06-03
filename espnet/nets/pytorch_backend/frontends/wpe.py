from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F
import torch_complex.functional as FC
from torch_complex.tensor import ComplexTensor

""" WPE pytorch version: Ported from https://github.com/fgnt/nara_wpe
Many functions aren't enough tested"""


def signal_framing(signal: Union[torch.Tensor, ComplexTensor],
                   frame_length: int, frame_step: int,
                   pad_value=0) -> Union[torch.Tensor, ComplexTensor]:
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
    else:
        signal = F.pad(signal, (0, frame_length - 1), 'constant', pad_value)
        indices = sum([list(range(i, i + frame_length))
                       for i in range(0, signal.size(-1) - frame_length + 1,
                                      frame_step)], [])

        signal = signal[..., indices].view(*signal.size()[:-1], -1,
                                           frame_length)
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


def get_power_online(signal: ComplexTensor) -> torch.Tensor:
    """Calculates power for `signal`

    Args:
        signal : Single frequency signal
            with shape (F, C, T).
        axis: reduce_mean axis
    Returns:
        Power with shape (F, )

    """
    power = signal.real ** 2 + signal.imag ** 2
    power = power.mean(dim=-1).mean(dim=-2)
    return power


def get_correlations(Y: ComplexTensor, inverse_power: torch.Tensor,
                     taps, delay) -> Tuple[ComplexTensor, ComplexTensor]:
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
    assert inverse_power.size(0) == Y.size(0), \
        (inverse_power.size(0), Y.size(0))

    F, C, T = Y.size()

    # Y: (F, C, T) -> Psi: (F, C, T, taps)
    Psi = signal_framing(
        Y, frame_length=taps, frame_step=1)[..., :T - delay - taps + 1, :]
    # Reverse along taps-axis
    Psi = FC.reverse(Psi, dim=-1)
    Psi_conj_norm = \
        Psi.conj() * inverse_power[..., None, delay + taps - 1:, None]

    # (F, C, T, taps) x (F, C, T, taps) -> (F, taps, C, taps, C)
    correlation_matrix = FC.einsum('fdtk,fetl->fkdle', (Psi_conj_norm, Psi))
    # (F, taps, C, taps, C) -> (F, taps * C, taps * C)
    correlation_matrix = correlation_matrix.view(F, taps * C, taps * C)

    # (F, C, T, taps) x (F, C, T) -> (F, taps, C, C)
    correlation_vector = FC.einsum(
        'fdtk,fet->fked', (Psi_conj_norm, Y[..., delay + taps - 1:]))

    return correlation_matrix, correlation_vector


def get_filter_matrix_conj(correlation_matrix: ComplexTensor,
                           correlation_vector: ComplexTensor) -> ComplexTensor:
    """Calculate (conjugate) filter matrix based on correlations for one freq.

    Args:
        correlation_matrix : Correlation matrix (F, taps * C, taps * C)
        correlation_vector : Correlation vector (F, taps, C, C)

    Returns:
        filter_matrix_conj (ComplexTensor): (F, taps, C, C)
    """
    F, taps, C, _ = correlation_vector.size()

    # (F, taps, C1, C2) -> (F, C1, taps, C2) -> (F, C1, taps * C2)
    correlation_vector = \
        correlation_vector.permute(0, 2, 1, 3)\
        .contiguous().view(F, C, taps * C)

    inv_correlation_matrix = correlation_matrix.inverse()
    # (F, C, taps, C) x (F, taps * C, taps * C) -> (F, C, taps * C)
    stacked_filter_conj = FC.matmul(correlation_vector,
                                    inv_correlation_matrix.transpose(-1, -2))

    # (F, C1, taps * C2) -> (F, C1, taps, C2) -> (F, taps, C2, C1)
    filter_matrix_conj = \
        stacked_filter_conj.view(F, C, taps, C).permute(0, 2, 3, 1)
    return filter_matrix_conj


def perform_filter_operation(Y: ComplexTensor,
                             filter_matrix_conj: ComplexTensor, taps, delay) \
        -> ComplexTensor:
    """perform_filter_operation

    Args:
        Y : Complex-valued STFT signal of shape (F, C, T)
        filter Matrix (F, taps, C, C)
    """
    T = Y.size(-1)
    reverb_tail = ComplexTensor(torch.zeros_like(Y.real),
                                torch.zeros_like(Y.real))
    for tau_minus_delay in range(taps):
        new = FC.einsum('fde,fdt->fet',
                        (filter_matrix_conj[:, tau_minus_delay, :, :],
                         Y[:, :, :T - delay - tau_minus_delay]))
        new = FC.pad(new, (delay + tau_minus_delay, 0),
                     mode='constant', value=0)
        reverb_tail = reverb_tail + new

    return Y - reverb_tail


def perform_filter_operation_v2(Y: ComplexTensor,
                                filter_matrix_conj: ComplexTensor,
                                taps, delay) -> ComplexTensor:
    """perform_filter_operation_v2

    Args:
        Y : Complex-valued STFT signal of shape (F, C, T)
        filter Matrix (F, taps, C, C)
    """
    T = Y.size(-1)
    # Y_tilde: (taps, F, C, T)
    Y_tilde = FC.stack([FC.pad(Y[:, :, :T - delay - i], (delay + i, 0),
                               mode='constant', value=0)
                        for i in range(taps)],
                       dim=0)
    reverb_tail = FC.einsum('fpde,pfdt->fet', (filter_matrix_conj, Y_tilde))
    return Y - reverb_tail


def wpe_one_iteration(Y: ComplexTensor,
                      power: torch.Tensor,
                      taps: int = 10,
                      delay: int = 3,
                      eps: float = 1e-10,
                      inverse_power: bool = True) -> ComplexTensor:
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

    correlation_matrix, correlation_vector = \
        get_correlations(Y, inverse_power, taps, delay)
    filter_matrix_conj = get_filter_matrix_conj(
        correlation_matrix, correlation_vector)
    enhanced = perform_filter_operation_v2(Y, filter_matrix_conj, taps, delay)

    enhanced = enhanced.view(*batch_freq_size, *Y.size()[-2:])
    return enhanced


def wpe(Y: ComplexTensor, taps=10, delay=3, iterations=3) -> ComplexTensor:
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


def online_wpe_step(
        input_buffer: ComplexTensor,
        power: torch.Tensor,
        inv_cov: ComplexTensor = None,
        filter_taps: ComplexTensor = None,
        alpha: float = 0.99,
        taps: int = 10,
        delay: int = 3):
    """One step of online dereverberation.

    Args:
        input_buffer: (F, C, taps + delay + 1)
        power: Estimate for the current PSD (F, T)
        inv_cov: Current estimate of R^-1
        filter_taps: Current estimate of filter taps (F, taps * C, taps)
        alpha (float): Smoothing factor
        taps (int): Number of filter taps
        delay (int): Delay in frames

    Returns:
        Dereverberated frame of shape (F, D)
        Updated estimate of R^-1
        Updated estimate of the filter taps


    >>> frame_length = 512
    >>> frame_shift = 128
    >>> taps = 6
    >>> delay = 3
    >>> alpha = 0.999
    >>> frequency_bins = frame_length // 2 + 1
    >>> Q = None
    >>> G = None
    >>> unreverbed, Q, G = online_wpe_step(stft, get_power_online(stft), Q, G,
    ...                                    alpha=alpha, taps=taps, delay=delay)

    """
    assert input_buffer.size(-1) == taps + delay + 1, input_buffer.size()
    C = input_buffer.size(-2)

    if inv_cov is None:
        inv_cov = ComplexTensor(
            torch.eye(C * taps, dtype=input_buffer.dtype).expand(
                *input_buffer.size()[:-2], C * taps, C * taps))
    if filter_taps is None:
        filter_taps = ComplexTensor(
            torch.zeros(*input_buffer.size()[:-2], C * taps, C,
                        dtype=input_buffer.dtype))

    window = FC.reverse(input_buffer[..., :-delay - 1], dim=-1)
    # (..., C, T) -> (..., C * T)
    window = window.view(*input_buffer.size()[:-2], -1)
    pred = input_buffer[..., -1] - FC.einsum('...id,...i->...d',
                                             (filter_taps.conj(), window))

    nominator = FC.einsum('...ij,...j->...i', (inv_cov, window))
    denominator = \
        FC.einsum('...i,...i->...', (window.conj(), nominator)) + alpha * power
    kalman_gain = nominator / denominator[..., None]

    inv_cov_k = inv_cov - FC.einsum(
        '...j,...jm,...i->...im', (window.conj(), inv_cov, kalman_gain))
    inv_cov_k /= alpha

    filter_taps_k = \
        filter_taps + FC.einsum('...i,...m->...im', (kalman_gain, pred.conj()))
    return pred, inv_cov_k, filter_taps_k
