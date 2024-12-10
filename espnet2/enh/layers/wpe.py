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
    """
    Expands a signal into overlapping frames for further processing.

This function is part of the WPE (Weighted Prediction Error) algorithm, which is
used in audio signal processing, particularly in enhancing speech signals.

Attributes:
    signal (Union[torch.Tensor, ComplexTensor]): The input signal to be framed.
    frame_length (int): The length of each frame.
    frame_step (int): The step size between frames.
    pad_value (int, optional): The value to pad the signal with. Defaults to 0.

Args:
    signal: A tensor of shape (B * F, D, T), where B is the batch size,
        F is the number of frequency bins, D is the number of channels, and T
        is the number of time steps. Can be a real or complex tensor.
    frame_length: The length of each frame to create from the signal.
    frame_step: The number of time steps to step between frames.
    pad_value: The value used to pad the signal if it is shorter than required
        for framing.

Returns:
    Union[torch.Tensor, ComplexTensor]: A tensor of shape (B * F, D, T, W),
    where W is the number of frames created from the signal.

Examples:
    >>> import torch
    >>> signal = torch.randn(2, 1, 10)  # Example signal of shape (B, D, T)
    >>> framed_signal = signal_framing(signal, frame_length=4, frame_step=2)
    >>> print(framed_signal.shape)  # Output shape should be (2, 1, 4, 4)

Note:
    This function supports both real-valued and complex-valued signals. If the
    input signal is complex, it will frame both the real and imaginary parts
    separately and return a ComplexTensor.

Todo:
    - Improve error handling for incompatible signal shapes.
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
    """
    Calculates power for a given signal.

The power is calculated as the mean of the squared real and imaginary parts 
of the signal across the specified dimension.

Args:
    signal (Union[torch.Tensor, ComplexTensor]): 
        Single frequency signal with shape (F, C, T), where F is the number of 
        frequency bins, C is the number of channels, and T is the number of time 
        steps.
    dim (int, optional): 
        The dimension along which to compute the mean. Defaults to -2.

Returns:
    torch.Tensor: 
        Power with shape (F, T), representing the average power of the signal 
        across the specified dimension.

Examples:
    >>> signal = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 1, 1], [1, 1, 1]]])
    >>> power = get_power(signal)
    >>> print(power)
    tensor([[  4.5000,   5.5000,   6.5000],
            [  1.0000,   1.0000,   1.0000]])
    """
    power = signal.real**2 + signal.imag**2
    power = power.mean(dim=dim)
    return power


def get_correlations(
    Y: Union[torch.Tensor, ComplexTensor], inverse_power: torch.Tensor, taps, delay
) -> Tuple[Union[torch.Tensor, ComplexTensor], Union[torch.Tensor, ComplexTensor]]:
    """
    Calculates weighted correlations of a window of length taps.

This function computes the correlation matrix and correlation vector for a 
given complex-valued Short-Time Fourier Transform (STFT) signal using a 
weighted approach based on the provided inverse power and correlation 
parameters.

Attributes:
    Y (Union[torch.Tensor, ComplexTensor]): 
        Complex-valued STFT signal with shape (F, C, T).
    inverse_power (torch.Tensor): 
        Weighting factor with shape (F, T).
    taps (int): 
        Length of the correlation window.
    delay (int): 
        Delay for the weighting factor.

Args:
    Y: Union[torch.Tensor, ComplexTensor]
        Complex-valued STFT signal with shape (F, C, T).
    inverse_power: torch.Tensor
        Weighting factor with shape (F, T).
    taps: int
        Length of correlation window.
    delay: int
        Delay for the weighting factor.

Returns:
    Tuple[Union[torch.Tensor, ComplexTensor], Union[torch.Tensor, ComplexTensor]]:
        - Correlation matrix of shape (F, taps*C, taps*C).
        - Correlation vector of shape (F, taps, C, C).

Raises:
    AssertionError: If the dimensions of `inverse_power` do not match with 
    the dimensions of `Y`.

Examples:
    >>> Y = torch.randn(4, 2, 10, dtype=torch.complex64)  # (F, C, T)
    >>> inverse_power = torch.randn(4, 10)  # (F, T)
    >>> taps = 5
    >>> delay = 2
    >>> correlation_matrix, correlation_vector = get_correlations(Y, 
    ... inverse_power, taps, delay)
    >>> print(correlation_matrix.shape)  # (4, 10, 10)
    >>> print(correlation_vector.shape)   # (4, 5, 2, 2)

Note:
    This function assumes that the input tensors are properly shaped and 
    the operations will be performed in a batch manner.
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
    """
    Calculate (conjugate) filter matrix based on correlations for one frequency.

This function computes the conjugate filter matrix using the provided 
correlation matrix and correlation vector. The filter matrix is 
used in various signal processing applications, particularly in 
the context of enhancing audio signals.

Attributes:
    correlation_matrix (Union[torch.Tensor, ComplexTensor]): 
        Correlation matrix of shape (F, taps * C, taps * C).
    correlation_vector (Union[torch.Tensor, ComplexTensor]): 
        Correlation vector of shape (F, taps, C, C).
    eps (float): 
        A small value to ensure numerical stability when inverting 
        the correlation matrix.

Args:
    correlation_matrix: The correlation matrix from which the filter 
        matrix will be computed. Expected shape is (F, taps * C, 
        taps * C).
    correlation_vector: The correlation vector corresponding to 
        the correlation matrix. Expected shape is (F, taps, C, C).
    eps: A small constant added to the diagonal of the correlation 
        matrix to prevent singularity. Default is 1e-10.

Returns:
    Union[torch.Tensor, ComplexTensor]: The computed filter matrix 
    of shape (F, taps, C, C).

Examples:
    >>> correlation_matrix = torch.randn(5, 20, 20)  # F=5, taps*C=20
    >>> correlation_vector = torch.randn(5, 10, 4, 4)  # F=5, taps=10, C=4
    >>> filter_matrix_conj = get_filter_matrix_conj(correlation_matrix, 
    ... correlation_vector)
    >>> print(filter_matrix_conj.shape)  # Output: (5, 10, 4, 4)

Note:
    This function requires PyTorch version 1.9 or later for complex 
    tensor support. Ensure that the input tensors are of the correct 
    shape to avoid runtime errors.
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
    """
    Performs the filtering operation using the conjugate filter matrix.

    This function applies a filter to the input complex-valued STFT signal 
    using a conjugate filter matrix. The output is a modified signal with 
    reverberation reduced based on the filter parameters.

    Args:
        Y : Union[torch.Tensor, ComplexTensor]
            Complex-valued STFT signal of shape (F, C, T), where:
            - F is the number of frequency bins,
            - C is the number of channels, and
            - T is the number of time frames.
        filter_matrix_conj : Union[torch.Tensor, ComplexTensor]
            Conjugate filter matrix of shape (F, taps, C, C), where:
            - taps is the number of filter taps.
        taps : int
            Number of filter taps used in the filtering operation.
        delay : int
            Delay to be applied to the signal for the filtering operation.

    Returns:
        Union[torch.Tensor, ComplexTensor]
            The filtered signal of the same shape as input Y, 
            which is (F, C, T).

    Raises:
        ValueError: If the PyTorch version is lower than 1.9.0 and the input 
        signal Y is complex.

    Examples:
        >>> import torch
        >>> Y = torch.randn(5, 2, 100, dtype=torch.complex64)
        >>> filter_matrix_conj = torch.randn(5, 10, 2, 2, dtype=torch.complex64)
        >>> taps = 10
        >>> delay = 3
        >>> filtered_signal = perform_filter_operation(Y, filter_matrix_conj, taps, delay)
        >>> print(filtered_signal.shape)
        torch.Size([5, 2, 100])

    Note:
        Ensure that the input signal Y is compatible with the filter matrix 
        dimensions and the specified taps and delay.
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
    """
    WPE for one iteration.

    This function performs one iteration of the Weighted Prediction Error 
    (WPE) algorithm on a complex-valued Short-Time Fourier Transform (STFT) 
    signal. It enhances the input signal by applying a filter based on the 
    calculated correlations of the signal.

    Args:
        Y: Complex-valued STFT signal with shape (..., C, T).
        power: Power of the signal with shape (..., T).
        taps: Number of filter taps (default: 10).
        delay: Delay as a guard interval to prevent X from becoming zero 
            (default: 3).
        eps: Small value to prevent division by zero in the inverse power 
            calculation (default: 1e-10).
        inverse_power (bool): If True, uses the inverse of the power; 
            otherwise, uses the power itself (default: True).

    Returns:
        enhanced: Enhanced signal with shape (..., C, T).

    Raises:
        ValueError: If the input tensor `Y` does not match the shape of 
        `power` in the last dimension.

    Examples:
        >>> import torch
        >>> Y = torch.randn(1, 2, 100, dtype=torch.complex64)  # Shape (B, C, T)
        >>> power = torch.abs(Y)**2  # Calculate power
        >>> enhanced_signal = wpe_one_iteration(Y, power)
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
    """
    WPE (Weighted Prediction Error) for enhancing complex-valued STFT signals.

This implementation is a PyTorch version of WPE, originally ported from
https://github.com/fgnt/nara_wpe. The algorithm processes the input signal 
to enhance its quality by applying a series of filter operations based on 
weighted correlations.

Attributes:
    is_torch_1_9_plus (bool): Indicates if the PyTorch version is 1.9 or above.

Args:
    Y (Union[torch.Tensor, ComplexTensor]): 
        Complex-valued STFT signal with shape (F, C, T).
    taps (int): 
        Number of filter taps for the WPE algorithm. Default is 10.
    delay (int): 
        Delay as a guard interval to prevent the signal from becoming zero. 
        Default is 3.
    iterations (int): 
        Number of iterations to perform WPE. Default is 3.

Returns:
    Union[torch.Tensor, ComplexTensor]: 
        Enhanced signal with shape (F, C, T) after applying WPE.

Examples:
    >>> import torch
    >>> from espnet2.enh.layers.wpe import wpe
    >>> Y = torch.randn(64, 2, 100)  # Example STFT signal
    >>> enhanced_signal = wpe(Y, taps=10, delay=3, iterations=3)
    >>> print(enhanced_signal.shape)
    torch.Size([64, 2, 100])  # Enhanced signal shape is the same as input

Note:
    This function assumes that the input signal is properly formatted and 
    contains complex values.

Todo:
    - Implement additional tests to validate the performance of the WPE 
      algorithm.
    """
    enhanced = Y
    for _ in range(iterations):
        power = get_power(enhanced)
        enhanced = wpe_one_iteration(Y, power, taps=taps, delay=delay)
    return enhanced
