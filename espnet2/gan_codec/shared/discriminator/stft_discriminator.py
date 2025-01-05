# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HiFi-GAN Modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ModReLU(nn.Module):
    """
    Complex ReLU module.

    This module applies a modified ReLU activation function to complex-valued
    inputs. It computes the ReLU of the absolute value of the input, adds a
    learnable bias, and then reconstructs the complex output using the
    original phase of the input.

    Reference:
        - https://arxiv.org/abs/1705.09792
        - https://github.com/pytorch/pytorch/issues/47052#issuecomment-718948801

    Attributes:
        b (torch.Parameter): Learnable parameter that is added to the absolute
            value of the input before applying the ReLU function.

    Args:
        None

    Returns:
        Tensor: The complex output after applying the modified ReLU.

    Examples:
        >>> mod_relu = ModReLU()
        >>> input_tensor = torch.tensor([1+2j, -3-4j, 0+0j])
        >>> output_tensor = mod_relu(input_tensor)
        >>> print(output_tensor)
        tensor([3.0 + 2.0j, 0.0 + 0.0j, 0.0 + 0.0j])
    """

    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
            Calculate forward propagation.

        This method takes an input signal and processes it through the layers of
        the ComplexSTFT Discriminator. The signal undergoes a Short-Time Fourier
        Transform (STFT) and is then passed through a series of complex
        convolutional layers. The output can be either the absolute values of the
        logits or the real part of the complex output, depending on the
        configuration.

        Args:
            x (Tensor): Input signal of shape (B, 1, T), where B is the batch
            size, 1 is the number of input channels, and T is the length of the
            signal.

        Returns:
            List[List[Tensor]]: A nested list containing the discriminator output
            after processing through the network. The output is in the form of
            complex tensors.

        Reference:
            Paper: https://arxiv.org/pdf/2107.03312.pdf
            Implementation: https://github.com/alibaba-damo-academy/FunCodec.git

        Examples:
            >>> model = ComplexSTFTDiscriminator()
            >>> input_signal = torch.randn(8, 1, 1024)  # Batch of 8 signals
            >>> output = model(input_signal)
            >>> print(len(output))  # Should print 1
            >>> print(output[0][0].shape)  # Shape of the output tensor

        Note:
            Ensure that the input tensor `x` is appropriately shaped and contains
            the expected type of data (float or complex) for proper processing
            through the model.
        """
        return F.relu(torch.abs(x) + self.b) * torch.exp(1.0j * torch.angle(x))


class ComplexConv2d(nn.Module):
    """
        ComplexConv2d module that performs complex-valued 2D convolution.

    This module extends the standard 2D convolution to handle complex-valued
    inputs. The weights and biases are stored in real-valued format and are
    converted to complex format during the forward pass.

    Attributes:
        weight (torch.Tensor): The complex convolution weights as real-valued tensor.
        bias (torch.Tensor): The complex convolution biases as real-valued tensor.
        stride (int): The stride of the convolution.
        padding (int): The padding applied to the input.

    Args:
        dim (int): Number of input channels.
        dim_out (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolution kernel.
        stride (int or tuple, optional): Stride of the convolution. Default is 1.
        padding (int or tuple, optional): Padding added to both sides of the input.
            Default is 0.

    Returns:
        Tensor: The result of the complex 2D convolution.

    Examples:
        >>> import torch
        >>> conv = ComplexConv2d(dim=2, dim_out=3, kernel_size=3)
        >>> input_tensor = torch.randn(1, 2, 10, 10, dtype=torch.complex64)
        >>> output = conv(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 3, 8, 8])  # Output shape depends on padding and stride
    """

    def __init__(self, dim, dim_out, kernel_size, stride=1, padding=0):
        super().__init__()
        conv = nn.Conv2d(dim, dim_out, kernel_size, dtype=torch.complex64)
        self.weight = nn.Parameter(torch.view_as_real(conv.weight))
        self.bias = nn.Parameter(torch.view_as_real(conv.bias))

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
            Calculate forward propagation.

        This method processes the input signal through the Complex STFT
        Discriminator using complex convolutional layers and performs the
        Short-Time Fourier Transform (STFT) on the input.

        Args:
            x (Tensor): Input signal with shape (B, 1, T), where B is the
                batch size, and T is the length of the signal.

        Returns:
            List[List[Tensor]]: A nested list containing the output of the
                discriminator after processing the input signal through all
                layers.

        Reference:
            Paper: https://arxiv.org/pdf/2107.03312.pdf
            Implementation: https://github.com/alibaba-damo-academy/FunCodec.git

        Examples:
            >>> discriminator = ComplexSTFTDiscriminator()
            >>> input_signal = torch.randn(4, 1, 1024)  # Batch of 4 signals
            >>> output = discriminator(input_signal)
            >>> print(len(output))  # Should print the number of layers
            >>> print(output[0][0].shape)  # Shape of the output tensor
        """
        weight, bias = map(torch.view_as_complex, (self.weight, self.bias))

        x = x.to(weight.dtype)
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)


def ComplexSTFTResidualUnit(in_channel, out_channel, strides):
    """
        Complex STFT Residual block for building a discriminator in a GAN framework.

    This module serves as a building block for the ComplexSTFTDiscriminator,
    which processes complex-valued inputs and applies a series of complex
    convolutions and non-linear activations.

    Attributes:
        in_channel (int): Number of input channels for the convolutions.
        out_channel (int): Number of output channels after convolutions.
        strides (int): Stride size for convolutions.

    Args:
        in_channel (int): Input channel size.
        out_channel (int): Output channel size.
        strides (int): Stride size of the convolutions.

    Returns:
        nn.Module: A sequential module containing complex convolutions and
        ModReLU activation.

    Examples:
        >>> complex_stft_unit = ComplexSTFTResidualUnit(1, 2, (1, 2))
        >>> output_module = complex_stft_unit(torch.randn(1, 1, 64, 64))
        >>> print(output_module.shape)
        torch.Size([1, 2, 32, 32])
    """
    kernel_sizes = tuple(map(lambda t: t + 2, strides))
    paddings = tuple(map(lambda t: t // 2, kernel_sizes))

    return nn.Sequential(
        ComplexConv2d(in_channel, in_channel, 3, padding=1),
        ModReLU(),
        ComplexConv2d(
            in_channel, out_channel, kernel_sizes, stride=strides, padding=paddings
        ),
    )


class ComplexSTFTDiscriminator(nn.Module):
    """
    ComplexSTFT Discriminator used in SoundStream.

    This class implements a complex Short-Time Fourier Transform (STFT)
    discriminator for use in SoundStream. The architecture consists of
    several residual units with complex convolutional layers, allowing
    for effective processing of audio signals in the frequency domain.

    Adapted from https://github.com/alibaba-damo-academy/FunCodec.git.

    Attributes:
        init_conv (ComplexConv2d): Initial complex convolutional layer.
        layers (nn.ModuleList): List of complex STFT residual units.
        stft_normalized (bool): Flag to indicate if STFT output is normalized.
        logits_abs (bool): Flag to determine if the output logits are absolute.
        n_fft (int): FFT size used in STFT computation.
        hop_length (int): Hop length for STFT.
        win_length (int): Window length for STFT.

    Args:
        in_channels (int): Input channel (default: 1).
        channels (int): Number of output channels (default: 32).
        strides (List[List[int]]): Detailed strides for conv2d modules
            (default: [[1, 2], [2, 2], [1, 2], [2, 2], [1, 2], [2, 2]]).
        chan_mults (List[int]): Channel multipliers (default: [1, 2, 4, 4, 8, 8]).
        n_fft (int): n_fft in the STFT (default: 1024).
        hop_length (int): hop_length in the STFT (default: 256).
        win_length (int): win_length in the STFT (default: 1024).
        stft_normalized (bool): Whether to normalize the STFT output
            (default: False).
        logits_abs (bool): Whether to use the absolute number of output
            logits (default: True).

    Returns:
        None

    Examples:
        >>> discriminator = ComplexSTFTDiscriminator()
        >>> input_signal = torch.randn(1, 1, 16000)  # (B, C, T)
        >>> output = discriminator(input_signal)
        >>> print(len(output))  # Output: 1
        >>> print(output[0][0].shape)  # Shape of the discriminator output

    Note:
        The implementation is inspired by techniques from audio processing
        and the referenced papers.
    """

    def __init__(
        self,
        *,
        in_channels=1,
        channels=32,
        strides=[[1, 2], [2, 2], [1, 2], [2, 2], [1, 2], [2, 2]],
        chan_mults=[1, 2, 4, 4, 8, 8],
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        stft_normalized=False,
        logits_abs=True,
    ):
        """Initialize Complex STFT Discriminator used in SoundStream.

        Adapted from https://github.com/alibaba-damo-academy/FunCodec.git

        Args:
            in_channels (int): Input channel.
            channels (int): Output channel.
            strides (List[List(int, int)]): detailed strides in conv2d modules.
            chan_mults (List[int]): Channel multiplers.
            n_fft (int): n_fft in the STFT.
            hop_length (int): hop_length in the STFT.
            stft_normalized (bool): whether to normalize the stft output.
            logits_abs (bool): whether to use the absolute number of output logits.
        """
        super().__init__()
        self.init_conv = ComplexConv2d(in_channels, channels, 7, padding=3)

        layer_channels = tuple(map(lambda mult: mult * channels, chan_mults))
        layer_channels = (channels, *layer_channels)
        layer_channels_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        self.layers = nn.ModuleList([])

        for layer_stride, (in_channel, out_channel) in zip(
            strides, layer_channels_pairs
        ):
            self.layers.append(
                ComplexSTFTResidualUnit(in_channel, out_channel, layer_stride)
            )

        # stft settings
        self.stft_normalized = stft_normalized
        self.logits_abs = logits_abs

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x):
        """
                ComplexSTFTDiscriminator is a neural network module that implements a complex
        Short-Time Fourier Transform (STFT) discriminator used in SoundStream. It
        processes input signals to produce a list of outputs suitable for further
        discrimination tasks.

        Attributes:
            init_conv (ComplexConv2d): Initial complex convolution layer.
            layers (ModuleList): List of complex STFT residual units.
            stft_normalized (bool): Flag indicating whether to normalize the STFT output.
            logits_abs (bool): Flag indicating whether to return absolute logits.
            n_fft (int): FFT size for the STFT.
            hop_length (int): Hop length for the STFT.
            win_length (int): Window length for the STFT.

        Args:
            in_channels (int): Number of input channels (default: 1).
            channels (int): Number of output channels (default: 32).
            strides (List[List[int]]): Strides for the convolutional layers (default:
                [[1, 2], [2, 2], [1, 2], [2, 2], [1, 2], [2, 2]]).
            chan_mults (List[int]): Channel multipliers for each layer (default:
                [1, 2, 4, 4, 8, 8]).
            n_fft (int): FFT size for the STFT (default: 1024).
            hop_length (int): Hop length for the STFT (default: 256).
            win_length (int): Window length for the STFT (default: 1024).
            stft_normalized (bool): Whether to normalize the STFT output (default:
                False).
            logits_abs (bool): Whether to return absolute values of output logits
                (default: True).

        Returns:
            List[List[Tensor]]: A list of lists containing the discriminator output.

        Examples:
            # Example usage of ComplexSTFTDiscriminator
            discriminator = ComplexSTFTDiscriminator()
            input_signal = torch.randn(1, 1, 1024)  # Batch size of 1, 1 channel, 1024 time steps
            output = discriminator(input_signal)
            print(output)

        Note:
            This module is adapted from the implementation found at
            https://github.com/alibaba-damo-academy/FunCodec.git.

        Reference:
            Paper: https://arxiv.org/pdf/2107.03312.pdf
            Implementation: https://github.com/alibaba-damo-academy/FunCodec.git
        """
        x = x.squeeze(1)

        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            normalized=self.stft_normalized,
            return_complex=True,
        )

        x = rearrange(x, "b ... -> b 1 ...")

        x = self.init_conv(x)

        for layer in self.layers:
            x = layer(x)

        if self.logits_abs:
            x = torch.abs(x)
        else:
            x = torch.view_as_real(x)
        return [[x]]
