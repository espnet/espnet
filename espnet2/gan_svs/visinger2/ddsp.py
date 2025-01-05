import math

import librosa as li
import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
from scipy.signal import get_window


def safe_log(x):
    """
        Computes the safe logarithm of a tensor.

    This function applies the logarithm operation to the input tensor `x` while
    ensuring numerical stability by adding a small constant (1e-7) to `x`. This
    prevents issues that may arise from taking the logarithm of zero or negative
    values.

    Args:
        x (torch.Tensor): The input tensor for which the safe logarithm is to be
            computed.

    Returns:
        torch.Tensor: A tensor containing the safe logarithm of the input tensor.

    Examples:
        >>> import torch
        >>> x = torch.tensor([0.0, 1.0, 10.0])
        >>> safe_log(x)
        tensor([-7.0000e-08,  0.0000e+00,  2.3026e+00])

    Note:
        This function is particularly useful in scenarios where the input tensor
        may contain zeros, such as in probability distributions or feature
        transformations.
    """
    return torch.log(x + 1e-7)


@torch.no_grad()
def mean_std_loudness(dataset):
    """
    Calculate the mean and standard deviation of loudness values in a dataset.

    This function iterates over the dataset and computes the mean and standard
    deviation of the loudness values contained within. The dataset is expected
    to be an iterable containing tuples where the third element is a tensor
    representing loudness.

    Args:
        dataset (iterable): An iterable dataset where each element is a tuple
            containing data, labels, and loudness values (tensor).

    Returns:
        tuple: A tuple containing two elements:
            - mean (float): The computed mean of loudness values.
            - std (float): The computed standard deviation of loudness values.

    Examples:
        >>> dataset = [
        ...     (None, None, torch.tensor([0.1, 0.2, 0.3])),
        ...     (None, None, torch.tensor([0.2, 0.4, 0.6])),
        ...     (None, None, torch.tensor([0.5, 0.1, 0.3]))
        ... ]
        >>> mean, std = mean_std_loudness(dataset)
        >>> print(mean)
        0.3666666666666667
        >>> print(std)
        0.136164163677245
    """
    mean = 0
    std = 0
    n = 0
    for _, _, l in dataset:
        n += 1
        mean += (l.mean().item() - mean) / n
        std += (l.std().item() - std) / n
    return mean, std


def multiscale_fft(signal, scales, overlap):
    """
    Compute the multiscale Short-Time Fourier Transform (STFT) of a signal.

    This function calculates the STFT for a given signal across multiple scales
    specified in the `scales` list. The overlapping factor between consecutive
    frames is defined by the `overlap` parameter.

    Args:
        signal (torch.Tensor): Input audio signal tensor of shape (N, T),
            where N is the number of channels and T is the length of the signal.
        scales (list of int): List of scales (window sizes) for which the STFT
            will be computed.
        overlap (float): Fraction of overlap between consecutive frames, where
            0 < overlap < 1.

    Returns:
        list of torch.Tensor: A list containing the STFTs of the signal for
            each scale, where each tensor has shape (N, F, T'), F is the number
            of frequency bins, and T' is the number of time frames for the
            corresponding scale.

    Examples:
        >>> signal = torch.randn(1, 16000)  # Simulated audio signal
        >>> scales = [256, 512, 1024]
        >>> overlap = 0.5
        >>> stfts = multiscale_fft(signal, scales, overlap)
        >>> for idx, stft in enumerate(stfts):
        ...     print(f'STFT for scale {scales[idx]}: {stft.shape}')

    Note:
        The function uses a Hann window for the STFT computation. Ensure that
        the input signal is of type `torch.Tensor` and has the correct shape.
    """
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def resample(x, factor: int):
    """
        Resamples the input tensor by a given factor using a convolutional approach.

    This function takes a 3D tensor (batch, frame, channel) and resamples it by
    inserting zeros between the original samples, followed by a convolution with
    a Hann window to smooth the output.

    Args:
        x (torch.Tensor): A 3D tensor of shape (batch, frame, channel)
            representing the input signal to be resampled.
        factor (int): The factor by which to resample the input tensor.
            A factor of 2 will double the size of the output in the time dimension.

    Returns:
        torch.Tensor: A 3D tensor of shape (batch, factor * frame, channel)
            representing the resampled signal.

    Examples:
        >>> x = torch.rand(4, 100, 2)  # Batch of 4 signals, each with 100 frames and 2 channels
        >>> resampled = resample(x, 2)
        >>> print(resampled.shape)
        torch.Size([4, 200, 2])  # Output shape is now (batch, 200 frames, channel)

    Note:
        The input tensor should have at least 3 dimensions. The resampling
        is performed by inserting zeros between samples, followed by a
        convolution operation with a Hann window for smoothing.
    """
    batch, frame, channel = x.shape
    x = x.permute(0, 2, 1).reshape(batch * channel, 1, frame)

    window = torch.hann_window(
        factor * 2,
        dtype=x.dtype,
        device=x.device,
    ).reshape(1, 1, -1)
    y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2]).to(x)
    y[..., ::factor] = x
    y[..., -1:] = x[..., -1:]
    y = torch.nn.functional.pad(y, [factor, factor])
    y = torch.nn.functional.conv1d(y, window)[..., :-1]

    y = y.reshape(batch, channel, factor * frame).permute(0, 2, 1)

    return y


def upsample(signal, factor):
    """
    Upsample a given signal by a specified factor using interpolation.

    This function takes a 3-dimensional input tensor representing a batch of
    signals, where the dimensions correspond to (batch_size, num_channels,
    num_frames). It uses the PyTorch `interpolate` function to upsample the
    signal along the last dimension by the specified factor.

    Args:
        signal (torch.Tensor): Input tensor of shape (batch_size, num_channels,
            num_frames) representing the signal to be upsampled.
        factor (int): The upsampling factor. This determines how many times the
            signal will be upsampled.

    Returns:
        torch.Tensor: The upsampled signal tensor of shape (batch_size,
            num_channels, num_frames * factor).

    Examples:
        >>> import torch
        >>> signal = torch.randn(2, 3, 4)  # Example input tensor
        >>> upsampled_signal = upsample(signal, 2)
        >>> print(upsampled_signal.shape)
        torch.Size([2, 3, 8])  # Output shape is doubled in the last dimension

    Note:
        The upsampling is performed using linear interpolation, which may
        introduce artifacts in the signal. It is recommended to use this
        function when the input signal is well-sampled.
    """
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    """
    Remove frequencies above the Nyquist limit from the amplitude spectrum.

    The Nyquist limit is defined as half the sampling rate. This function
    multiplies the amplitudes of harmonics whose corresponding pitches are
    below the Nyquist frequency by 1, while setting the amplitudes of
    harmonics whose pitches are above the Nyquist frequency to a small
    value (1e-4).

    Args:
        amplitudes (torch.Tensor): A tensor containing the amplitudes of the
            harmonics. The shape should be (..., n_harm), where n_harm is
            the number of harmonics.
        pitch (torch.Tensor): A tensor containing the fundamental frequency
            (pitch) for each harmonic. The shape should be (...,).
        sampling_rate (float): The sampling rate of the signal.

    Returns:
        torch.Tensor: A tensor with the same shape as `amplitudes`, where
        amplitudes corresponding to harmonics above the Nyquist frequency
        have been reduced.

    Examples:
        >>> import torch
        >>> amplitudes = torch.tensor([[0.5, 0.3, 0.2]])
        >>> pitch = torch.tensor([440.0])
        >>> sampling_rate = 1000.0
        >>> result = remove_above_nyquist(amplitudes, pitch, sampling_rate)
        >>> print(result)
        tensor([[0.5000, 0.3000, 0.0001]])

    Note:
        The function assumes that the last dimension of `amplitudes`
        corresponds to the harmonics and that `pitch` is appropriately
        broadcastable with `amplitudes`.
    """
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa


def scale_function(x):
    """
    Scale the input tensor using a sigmoid function.

    This function applies a scaling transformation to the input tensor `x`
    using the sigmoid function. The output is adjusted to a specific range
    and includes a small constant to avoid numerical issues.

    Args:
        x (torch.Tensor): The input tensor to be scaled. It should have a
            shape compatible with the sigmoid function.

    Returns:
        torch.Tensor: The scaled output tensor, with the same shape as the
            input tensor.

    Examples:
        >>> import torch
        >>> x = torch.tensor([0.0, 1.0, 2.0])
        >>> scaled_x = scale_function(x)
        >>> print(scaled_x)
        tensor([1.0000e-07, 1.0000e+00, 1.0000e+00])

    Note:
        The output values are in the range of approximately [1e-7, 2] due
        to the nature of the sigmoid function and the scaling factor used.
    """
    return 2 * torch.sigmoid(x) ** (math.log(10)) + 1e-7


def extract_loudness(signal, sampling_rate, block_size, n_fft=2048):
    """
        Extracts the loudness of an audio signal using Short-Time Fourier Transform (STFT).

    This function computes the loudness of an input audio signal by applying the
    STFT and then applying A-weighting to the frequency bins. The resulting loudness
    is returned as a numpy array.

    Args:
        signal (np.ndarray): Input audio signal as a 1D numpy array.
        sampling_rate (int): The sampling rate of the audio signal in Hz.
        block_size (int): The number of samples per block for the STFT.
        n_fft (int, optional): The number of FFT points. Defaults to 2048.

    Returns:
        np.ndarray: The computed loudness of the audio signal as a 1D numpy array.

    Examples:
        >>> import numpy as np
        >>> signal = np.random.randn(44100)  # 1 second of random noise
        >>> loudness = extract_loudness(signal, 44100, 1024)
        >>> print(loudness.shape)  # Should output the shape of the loudness array

    Note:
        The loudness values are computed in decibels (dB) relative to the reference
        level.

    Raises:
        ValueError: If the signal is not a 1D numpy array or if the sampling rate
        is not a positive integer.
    """
    S = li.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    S = np.log(abs(S) + 1e-7)
    f = li.fft_frequencies(sampling_rate, n_fft)
    a_weight = li.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)

    S = np.mean(S, 0)[..., :-1]

    return S


# TODO(Yifeng): Some functions are not used here such as crepe,
#  maybe we can remove them later or only import used functions.
def extract_pitch(signal, sampling_rate, block_size):
    """
        Extract the fundamental frequency (pitch) from an audio signal using the CREPE
    algorithm.

    This function computes the pitch of an audio signal by utilizing the CREPE
    algorithm. It processes the input signal in blocks, extracting the fundamental
    frequency at each block. The resulting pitch is returned as a 1D numpy array.

    Attributes:
        signal (torch.Tensor): The input audio signal as a 1D tensor.
        sampling_rate (int): The sampling rate of the audio signal.
        block_size (int): The size of each block for processing the audio signal.

    Args:
        signal (torch.Tensor): The input audio signal from which to extract the pitch.
        sampling_rate (int): The sampling rate of the audio signal.
        block_size (int): The size of the block to divide the signal into for pitch
                          extraction.

    Returns:
        numpy.ndarray: A 1D array containing the extracted pitch values for each
                       block of the audio signal.

    Raises:
        ValueError: If the input signal is not a 1D tensor or if the sampling rate
                    or block size is non-positive.

    Examples:
        >>> import torch
        >>> signal = torch.randn(16000)  # Simulated 1-second audio signal
        >>> sampling_rate = 16000  # 16 kHz
        >>> block_size = 512  # Block size for processing
        >>> pitch = extract_pitch(signal, sampling_rate, block_size)
        >>> print(pitch.shape)  # Should print the shape of the pitch array

    Note:
        The function uses the CREPE algorithm for pitch extraction. Ensure that the
        necessary dependencies for CREPE are installed and available in your
        environment.

    Todo:
        - Consider implementing additional pitch extraction methods for
          comparison.
    """
    length = signal.shape[-1] // block_size
    f0 = crepe.predict(  # noqa
        signal,
        sampling_rate,
        step_size=int(1000 * block_size / sampling_rate),
        verbose=1,
        center=True,
        viterbi=True,
    )
    f0 = f0[1].reshape(-1)[:-1]

    if f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )

    return f0


def mlp(in_size, hidden_size, n_layers):
    """
    Construct a multi-layer perceptron (MLP) model.

    This function creates a multi-layer perceptron consisting of
    fully connected layers followed by layer normalization and
    LeakyReLU activation functions. The architecture of the MLP is
    defined by the number of input features, the size of hidden
    layers, and the number of layers.

    Args:
        in_size (int): The number of input features for the MLP.
        hidden_size (int): The number of neurons in each hidden layer.
        n_layers (int): The total number of hidden layers in the MLP.

    Returns:
        nn.Sequential: A PyTorch sequential model representing the MLP.

    Examples:
        >>> model = mlp(10, 20, 3)
        >>> print(model)
        Sequential(
          (0): Linear(in_features=10, out_features=20, bias=True)
          (1): LayerNorm((20,), eps=1e-05, elementwise_affine=True)
          (2): LeakyReLU(negative_slope=0.01)
          (3): Linear(in_features=20, out_features=20, bias=True)
          (4): LayerNorm((20,), eps=1e-05, elementwise_affine=True)
          (5): LeakyReLU(negative_slope=0.01)
          (6): Linear(in_features=20, out_features=20, bias=True)
          (7): LayerNorm((20,), eps=1e-05, elementwise_affine=True)
          (8): LeakyReLU(negative_slope=0.01)
        )
    """
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def gru(n_input, hidden_size):
    """
    Create a Gated Recurrent Unit (GRU) layer.

    This function constructs a GRU layer using PyTorch's `nn.GRU`. The GRU
    layer is a type of recurrent neural network layer that is well-suited for
    sequence modeling tasks, providing a mechanism to retain information over
    long sequences while addressing the vanishing gradient problem.

    Args:
        n_input (int): The number of input features for each time step.
        hidden_size (int): The number of features in the hidden state.

    Returns:
        nn.GRU: A GRU layer configured with the specified input size and
        hidden size.

    Examples:
        >>> gru_layer = gru(n_input=10, hidden_size=20)
        >>> input_tensor = torch.randn(5, 3, 10)  # (batch_size, seq_len, n_input)
        >>> output, hidden = gru_layer(input_tensor)
        >>> output.shape
        torch.Size([5, 3, 20])  # (batch_size, seq_len, hidden_size)
    """
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)


def harmonic_synth(pitch, amplitudes, sampling_rate):
    """
    Generate a harmonic signal based on pitch and amplitude information.

    This function synthesizes a harmonic signal by computing the sum of
    sinusoidal waves, where each harmonic is weighted by the provided
    amplitude values. The pitch determines the fundamental frequency,
    and the number of harmonics is determined by the length of the
    amplitudes tensor.

    Args:
        pitch (torch.Tensor): A tensor containing the fundamental frequency
            (in Hz) for each time step. Shape should be (batch_size,).
        amplitudes (torch.Tensor): A tensor containing the amplitudes for
            each harmonic. Shape should be (batch_size, n_harmonics).
        sampling_rate (int): The sampling rate of the generated signal.

    Returns:
        torch.Tensor: A tensor containing the synthesized harmonic signal.
            Shape will be (batch_size, 1, signal_length).

    Examples:
        >>> pitch = torch.tensor([[440.0]])  # A4 note
        >>> amplitudes = torch.tensor([[1.0, 0.5, 0.25]])  # Amplitudes for harmonics
        >>> sampling_rate = 44100  # Standard audio sampling rate
        >>> signal = harmonic_synth(pitch, amplitudes, sampling_rate)
        >>> print(signal.shape)  # Should output: torch.Size([1, 1, signal_length])

    Note:
        The output signal will be a single channel (mono) audio signal
        that can be played or processed further.
    """
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal


def amp_to_impulse_response(amp, target_size):
    """
        Converts an amplitude spectrum to an impulse response using the inverse FFT.

    This function takes an amplitude spectrum and converts it into an impulse
    response suitable for signal processing tasks. It first stacks the amplitude
    spectrum with a zero imaginary part, converts it to a complex tensor, and
    applies the inverse real FFT to obtain the time-domain signal. The result
    is then windowed and padded to match the specified target size.

    Args:
        amp (torch.Tensor): The input amplitude spectrum of shape (n_freq,).
        target_size (int): The desired size of the output impulse response.

    Returns:
        torch.Tensor: The impulse response of shape (target_size,).

    Examples:
        >>> amp = torch.tensor([0.1, 0.2, 0.3, 0.4])
        >>> target_size = 8
        >>> impulse_response = amp_to_impulse_response(amp, target_size)
        >>> print(impulse_response.shape)
        torch.Size([8])

    Note:
        The input amplitude spectrum should contain non-negative values, and the
        target size must be greater than or equal to the size of the computed
        impulse response.

    Raises:
        ValueError: If the target_size is less than the length of the computed
        impulse response.
    """
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    """
    Perform convolution of a signal with a kernel using FFT.

    This function takes an input signal and a kernel, both of which are
    expected to be 1D tensors. The convolution is performed in the frequency
    domain using Fast Fourier Transform (FFT) for efficient computation. The
    output signal is obtained by applying the inverse FFT to the product of
    the FFTs of the signal and the kernel.

    Args:
        signal (torch.Tensor): The input signal tensor of shape (N, L), where
            N is the batch size and L is the length of the signal.
        kernel (torch.Tensor): The convolution kernel tensor of shape (M, K),
            where M is the batch size (should match signal's batch size)
            and K is the length of the kernel.

    Returns:
        torch.Tensor: The output tensor after convolution, with shape
            (N, L + K - 1), where the length of the output is the sum of the
            lengths of the signal and kernel minus one.

    Examples:
        >>> signal = torch.tensor([[1.0, 2.0, 3.0]])
        >>> kernel = torch.tensor([[0.5, 1.0]])
        >>> output = fft_convolve(signal, kernel)
        >>> print(output)
        tensor([[1.5000, 3.0000, 4.5000, 3.0000]])

    Note:
        The function applies zero-padding to both the input signal and the
        kernel to ensure that the convolution can be computed correctly
        without any boundary effects.

    Raises:
        ValueError: If the batch sizes of signal and kernel do not match.
    """
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2 :]

    return output


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    """
        Initializes the kernels for the signal processing tasks using Fourier basis.

    This function creates a kernel based on the specified window length, window
    increment, FFT length, and window type. It generates a Fourier basis and can
    optionally compute the pseudoinverse of the kernel.

    Args:
        win_len (int): The length of the window to be used for processing.
        win_inc (int): The increment between consecutive windows (not used in this
            implementation but may be relevant for future extensions).
        fft_len (int): The length of the FFT to be used.
        win_type (str, optional): The type of window to apply (e.g., 'hann',
            'hamming'). If None, a rectangular window is used.
        invers (bool, optional): If True, computes the pseudoinverse of the kernel.
            Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - A tensor representing the initialized kernel of shape
              (2 * win_len, 1, win_len).
            - A tensor representing the window of shape (1, win_len, 1).

    Examples:
        >>> kernel, window = init_kernels(win_len=256, win_inc=128, fft_len=512,
        ...                                win_type='hann')
        >>> kernel.shape
        torch.Size([512, 1, 256])
        >>> window.shape
        torch.Size([1, 256, 1])

    Note:
        The output kernel is used in signal processing tasks, such as convolution
        with signals in the time domain. The choice of window type can significantly
        affect the performance of subsequent processing.

    Todo:
        Consider implementing additional window types and handling for win_inc in
        future versions.
    """
    if win_type == "None" or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)  # **0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(
        window[None, :, None].astype(np.float32)
    )
