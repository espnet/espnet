# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/encodec/tree/main


import typing as tp
from abc import ABC, abstractmethod

import torch
import torchaudio
from einops import rearrange
from torch import nn

from espnet2.gan_codec.shared.discriminator.msstft_conv import NormConv2d


def get_2d_padding(
    kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)
):
    """
        Calculate the 2D padding required for a convolutional layer given the kernel
    size and dilation.

    The padding is computed to ensure that the output dimensions remain the same
    as the input dimensions when using the specified kernel size and dilation
    in a convolution operation.

    Args:
        kernel_size (tuple of int): The size of the convolution kernel in the
            format (height, width).
        dilation (tuple of int, optional): The dilation rate for the kernel in
            the format (dilation_height, dilation_width). Defaults to (1, 1).

    Returns:
        tuple: A tuple containing the calculated padding values in the format
            (padding_height, padding_width).

    Examples:
        >>> get_2d_padding((3, 3))
        (1, 1)

        >>> get_2d_padding((5, 5), dilation=(2, 2))
        (4, 4)

    Note:
        The padding values are computed using the formula:
        padding = ((kernel_size - 1) * dilation) // 2
    """
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class DiscriminatorSTFT(nn.Module):
    """
    STFT sub-discriminator for evaluating audio signals.

    This class implements a short-time Fourier transform (STFT) based
    discriminator that processes audio input through multiple convolutional
    layers. It is designed to work as part of a generative adversarial
    network (GAN) setup, where it assesses the quality of generated audio
    signals.

    Attributes:
        filters (int): Number of filters in convolutions.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_fft (int): Size of FFT for each scale.
        hop_length (int): Length of hop between STFT windows for each scale.
        win_length (int): Window size for each scale.
        normalized (bool): Whether to normalize by magnitude after STFT.
        activation (callable): Activation function used in the convolutional layers.
        convs (ModuleList): List of convolutional layers for feature extraction.
        conv_post (NormConv2d): Final convolutional layer to output logits.

    Args:
        filters (int): Number of filters in convolutions.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_fft (int): Size of FFT for each scale.
        hop_length (int): Length of hop between STFT windows for each scale.
        kernel_size (tuple of int): Inner Conv2d kernel sizes.
        stride (tuple of int): Inner Conv2d strides.
        dilations (list of int): Inner Conv2d dilation on the time dimension.
        win_length (int): Window size for each scale.
        normalized (bool): Whether to normalize by magnitude after STFT.
        norm (str): Normalization method.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters.

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor]]: A tuple containing the output
        logits and a list of feature maps from each layer.

    Examples:
        >>> discriminator = DiscriminatorSTFT(filters=64, in_channels=1)
        >>> input_tensor = torch.randn(1, 1, 16000)  # Example audio input
        >>> output, feature_maps = discriminator(input_tensor)
        >>> print(output.shape)  # Output logits shape
        torch.Size([1, 1, H, W])  # H and W depend on the input and config

    Note:
        Ensure the input tensor is of the correct shape (batch_size,
        in_channels, audio_length) before passing to the forward method.

    Todo:
        - Add support for different normalization methods.
        - Optimize the performance of the forward method.
    """

    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tp.Tuple[int, int] = (3, 9),
        dilations: tp.List = [1, 2, 4],
        stride: tp.Tuple[int, int] = (1, 2),
        normalized: bool = True,
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: dict = {"negative_slope": 0.2},
    ):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None,
        )
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                spec_channels,
                self.filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size),
            )
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm,
                )
            )
            in_chs = out_chs
        out_chs = min(
            (filters_scale ** (len(dilations) + 1)) * self.filters, max_filters
        )
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm,
            )
        )
        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm=norm,
        )

    def forward(self, x: torch.Tensor):
        """
            Forward pass for the STFT discriminator.

        This method processes the input tensor through the series of convolutional
        layers defined in the DiscriminatorSTFT class. It transforms the input
        using the Short-Time Fourier Transform (STFT), applies a series of
        convolutions, and collects feature maps at each stage. The output is a
        tuple containing the final output and the list of feature maps.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T] where:
                - B is the batch size,
                - C is the number of input channels,
                - T is the length of the input signal.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor after processing through the
                  convolutional layers.
                - list: A list of feature maps collected at each layer.

        Examples:
            >>> discriminator = DiscriminatorSTFT(filters=64)
            >>> input_tensor = torch.randn(8, 1, 1024)  # Batch of 8, 1 channel, 1024 length
            >>> output, feature_maps = discriminator(input_tensor)
            >>> print(output.shape)  # Output shape will depend on the architecture
            >>> print(len(feature_maps))  # Number of feature maps collected

        Note:
            Ensure that the input tensor is properly shaped and normalized
            as per the requirements of the STFT.

        Raises:
            ValueError: If the input tensor shape is not compatible with the
            expected dimensions.
        """
        fmap = []
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, "b c w t -> b c t w")
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiDiscriminator(ABC, nn.Module):
    """
    Base implementation for discriminators composed of sub-discriminators
    acting at different scales.

    This class serves as a base for implementing multi-scale discriminators
    that utilize several sub-discriminators to analyze input signals at
    different frequency scales. It defines the interface that derived
    classes must implement.

    Attributes:
        None

    Args:
        None

    Returns:
        None

    Yields:
        None

    Raises:
        NotImplementedError: If the `forward` method is not implemented in a
            derived class.
        NotImplementedError: If the `num_discriminators` property is not
            implemented in a derived class.

    Examples:
        To create a custom multi-discriminator, derive from this class and
        implement the `forward` method and the `num_discriminators` property:

        ```python
        class CustomMultiDiscriminator(MultiDiscriminator):
            def forward(self, x: torch.Tensor):
                # Custom forward implementation
                pass

            @property
            def num_discriminators(self) -> int:
                return 2  # Example number of discriminators
        ```

    Note:
        This is an abstract class and cannot be instantiated directly.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """
            Forward pass for the MultiScaleSTFTDiscriminator.

        This method processes the input tensor `x` through multiple sub-discriminators
        to produce outputs that capture various frequency and time scales. Each
        sub-discriminator computes its own feature maps and final logits.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T), where:
                - B: Batch size
                - C: Number of input channels
                - T: Length of the input sequence

        Returns:
            List[List[torch.Tensor]]: A list containing feature maps and logits from
            each sub-discriminator. Each entry in the list corresponds to a
            sub-discriminator and contains:
                - Feature maps from each layer of the sub-discriminator
                - Final logits from the last layer of the sub-discriminator

        Examples:
            >>> discriminator = MultiScaleSTFTDiscriminator(filters=64)
            >>> input_tensor = torch.randn(8, 1, 16000)  # Batch of 8, 1 channel, 16000 samples
            >>> output = discriminator(input_tensor)
            >>> print(len(output))  # Should print the number of sub-discriminators
            >>> print(len(output[0]))  # Should print the number of feature maps + 1 for logits

        Note:
            The input tensor should be preprocessed to match the expected shape
            before passing it to this method. Each sub-discriminator processes the
            input independently and the outputs are collected into a list.

        Raises:
            RuntimeError: If the input tensor does not have the correct number of
            dimensions or if it contains invalid values.
        """
        raise NotImplementedError("forward method is not implemented")

    @property
    @abstractmethod
    def num_discriminators(self) -> int:
        """
            Base implementation for discriminators composed of sub-discriminators
        acting at different scales.

        This class serves as a template for creating multi-scale discriminators,
        which are composed of several sub-discriminators. Each sub-discriminator
        operates at a different scale, allowing for a more comprehensive analysis
        of the input data.

        Attributes:
            None

        Args:
            None

        Returns:
            None

        Yields:
            None

        Raises:
            NotImplementedError: If the `forward` method is not implemented.
            NotImplementedError: If the `num_discriminators` property is not
            implemented.

        Examples:
            To create a derived class from `MultiDiscriminator`, implement the
            `forward` method and the `num_discriminators` property.

            class MyDiscriminator(MultiDiscriminator):
                def forward(self, x: torch.Tensor):
                    # Custom forward implementation
                    pass

                @property
                def num_discriminators(self) -> int:
                    return 3  # Example number of discriminators

        Note:
            This class cannot be instantiated directly, as it is intended to be
            a base class for specific multi-discriminator implementations.

        Todo:
            Implement specific multi-discriminator logic in derived classes.
        """
        raise NotImplementedError("num_discriminators is not implemented")


class MultiScaleSTFTDiscriminator(MultiDiscriminator):
    """
        Multi-Scale STFT (MS-STFT) discriminator.

    This class implements a multi-scale Short-Time Fourier Transform (STFT)
    discriminator, which consists of multiple sub-discriminators operating
    at different scales. It can be used in Generative Adversarial Networks
    (GANs) to evaluate the quality of generated audio signals.

    Attributes:
        sep_channels (bool): If True, separate channels to distinct samples
            for stereo support.
        discriminators (nn.ModuleList): List of STFT discriminators for each
            scale.

    Args:
        filters (int): Number of filters in convolutions.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        sep_channels (bool): Separate channels to distinct samples for stereo
            support.
        n_ffts (Sequence[int]): Size of FFT for each scale.
        hop_lengths (Sequence[int]): Length of hop between STFT windows for
            each scale.
        win_lengths (Sequence[int]): Window size for each scale.
        **kwargs: Additional args for STFTDiscriminator.

    Examples:
        >>> discriminator = MultiScaleSTFTDiscriminator(filters=64)
        >>> input_tensor = torch.randn(1, 1, 16000)  # Example input
        >>> output = discriminator(input_tensor)
        >>> print(len(output))  # Number of scales (discriminators)

    Note:
        The input tensor shape is expected to be (batch_size, channels, time).

    Raises:
        AssertionError: If the lengths of n_ffts, hop_lengths, and
        win_lengths are not equal.
    """

    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        sep_channels: bool = False,
        n_ffts: tp.List[int] = [1024, 2048, 512],
        hop_lengths: tp.List[int] = [256, 512, 128],
        win_lengths: tp.List[int] = [1024, 2048, 512],
        **kwargs
    ):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.sep_channels = sep_channels
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorSTFT(
                    filters,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_fft=n_ffts[i],
                    win_length=win_lengths[i],
                    hop_length=hop_lengths[i],
                    **kwargs
                )
                for i in range(len(n_ffts))
            ]
        )

    @property
    def num_discriminators(self):
        """
            Multi-Scale STFT (MS-STFT) discriminator.

        This class implements a multi-scale discriminator that utilizes STFT-based
        sub-discriminators to analyze audio signals at various scales. Each
        sub-discriminator processes the input signal using Short-Time Fourier Transform
        (STFT) to extract features that are relevant for the task at hand, such as
        audio classification or generation.

        Args:
            filters (int): Number of filters in convolutions.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            sep_channels (bool): Separate channels to distinct samples for stereo support.
            n_ffts (Sequence[int]): Size of FFT for each scale.
            hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale.
            win_lengths (Sequence[int]): Window size for each scale.
            **kwargs: Additional args for STFTDiscriminator.

        Attributes:
            sep_channels (bool): Indicates if channels are separated for stereo support.
            discriminators (nn.ModuleList): List of STFT sub-discriminators.

        Returns:
            List of features extracted from each scale along with logits.

        Examples:
            >>> discriminator = MultiScaleSTFTDiscriminator(filters=64)
            >>> input_tensor = torch.randn(1, 1, 16000)  # Example input
            >>> output = discriminator(input_tensor)
            >>> len(output)  # Should return the number of scales

        Note:
            The number of discriminators is equal to the length of the
            `n_ffts`, `hop_lengths`, and `win_lengths` parameters.

        Raises:
            AssertionError: If the lengths of `n_ffts`, `hop_lengths`,
            and `win_lengths` do not match.
        """
        return len(self.discriminators)

    def _separate_channels(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        return x.view(-1, 1, T)

    def forward(self, x: torch.Tensor):
        """
            Forward pass for the MultiScaleSTFTDiscriminator.

        This method processes the input tensor `x` through each sub-discriminator,
        computes the STFT, and returns the feature maps along with the logits from
        each sub-discriminator.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T], where B is the batch
                size, C is the number of channels, and T is the number of time steps.

        Returns:
            list: A list containing the feature maps and logits from each sub-discriminator.
                Each element in the list corresponds to the output of a sub-discriminator,
                which includes feature maps and the final logit.

        Examples:
            >>> discriminator = MultiScaleSTFTDiscriminator(filters=64)
            >>> input_tensor = torch.randn(8, 1, 16000)  # Batch of 8, 1 channel, 16000 samples
            >>> outputs = discriminator(input_tensor)
            >>> for output in outputs:
            ...     print(len(output))  # Each output will have the feature maps and logit

        Note:
            The input tensor should be pre-processed appropriately before being passed
            to this method, ensuring it has the correct shape and data type.
        """

        ans = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            ans.append(fmap + [logit])

        return ans
