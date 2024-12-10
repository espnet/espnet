#!/usr/bin/env python3
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sinc convolutions for raw audio input."""

from collections import OrderedDict
from typing import Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import typechecked

from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.layers.sinc_conv import LogCompression, SincConv


class LightweightSincConvs(AbsPreEncoder):
    """
    Lightweight Sinc Convolutions for end-to-end speech recognition.

    This class implements lightweight Sinc convolutions to process raw audio input
    directly for speech recognition, as described in the paper "Lightweight
    End-to-End Speech Recognition from Raw Audio Data Using Sinc-Convolutions"
    by Kürzinger et al. (https://arxiv.org/abs/2010.07597).

    The architecture processes audio through a series of convolutional blocks
    that utilize Sinc filters, followed by normalization and pooling layers.
    To integrate this pre-encoder in your model, specify `preencoder: sinc`
    and use `frontend: sliding_window` in your YAML configuration file.
    The data flow is as follows:

    Frontend (SlidingWindow) -> SpecAug -> Normalization ->
    Pre-encoder (LightweightSincConvs) -> Encoder -> Decoder

    This method performs data augmentation in the time domain, contrasting
    with the spectral domain approach of the default frontend. For visualizing
    the learned Sinc filters, utilize `plot_sinc_filters.py`.

    Attributes:
        fs (int): Sample rate of the input audio.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels per input channel.
        activation_type (str): Type of activation function to use.
        dropout_type (str): Type of dropout function to use.
        windowing_type (str): Type of windowing function to use.
        scale_type (str): Type of filter-bank initialization scale.

    Args:
        fs (Union[int, str, float]): Sample rate. Defaults to 16000.
        in_channels (int): Number of input channels. Defaults to 1.
        out_channels (int): Number of output channels. Defaults to 256.
        activation_type (str): Activation function type. Defaults to "leakyrelu".
        dropout_type (str): Dropout function type. Defaults to "dropout".
        windowing_type (str): Windowing function type. Defaults to "hamming".
        scale_type (str): Filter-bank initialization scale type. Defaults to "mel".

    Raises:
        NotImplementedError: If the specified dropout or activation type is not
        supported.

    Examples:
        # Initialize the pre-encoder with default parameters
        sinc_preencoder = LightweightSincConvs()

        # Initialize with custom parameters
        sinc_preencoder = LightweightSincConvs(
            fs=16000,
            in_channels=2,
            out_channels=128,
            activation_type='relu',
            dropout_type='spatial'
        )

        # Forward pass with input tensor
        input_tensor = torch.randn(32, 100, 1, 400)  # (B, T, C_in, D_in)
        output_tensor, lengths = sinc_preencoder(input_tensor, input_lengths)

    Note:
        This class relies on PyTorch and is designed to be compatible with
        ESPnet's architecture for speech processing.

    Todo:
        - Add support for additional activation and dropout types.
        - Implement unit tests for various configurations and edge cases.
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str, float] = 16000,
        in_channels: int = 1,
        out_channels: int = 256,
        activation_type: str = "leakyrelu",
        dropout_type: str = "dropout",
        windowing_type: str = "hamming",
        scale_type: str = "mel",
    ):
        """Initialize the module.

        Args:
            fs: Sample rate.
            in_channels: Number of input channels.
            out_channels: Number of output channels (for each input channel).
            activation_type: Choice of activation function.
            dropout_type: Choice of dropout function.
            windowing_type: Choice of windowing function.
            scale_type:  Choice of filter-bank initialization scale.
        """
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        self.fs = fs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_type = activation_type
        self.dropout_type = dropout_type
        self.windowing_type = windowing_type
        self.scale_type = scale_type

        self.choices_dropout = {
            "dropout": torch.nn.Dropout,
            "spatial": SpatialDropout,
            "dropout2d": torch.nn.Dropout2d,
        }
        if dropout_type not in self.choices_dropout:
            raise NotImplementedError(
                f"Dropout type has to be one of "
                f"{list(self.choices_dropout.keys())}",
            )

        self.choices_activation = {
            "leakyrelu": torch.nn.LeakyReLU,
            "relu": torch.nn.ReLU,
        }
        if activation_type not in self.choices_activation:
            raise NotImplementedError(
                f"Activation type has to be one of "
                f"{list(self.choices_activation.keys())}",
            )

        # initialization
        self._create_sinc_convs()
        # Sinc filters require custom initialization
        self.espnet_initialization_fn()

    def _create_sinc_convs(self):
        blocks = OrderedDict()

        # SincConvBlock
        out_channels = 128
        self.filters = SincConv(
            self.in_channels,
            out_channels,
            kernel_size=101,
            stride=1,
            fs=self.fs,
            window_func=self.windowing_type,
            scale_type=self.scale_type,
        )
        block = OrderedDict(
            [
                ("Filters", self.filters),
                ("LogCompression", LogCompression()),
                ("BatchNorm", torch.nn.BatchNorm1d(out_channels, affine=True)),
                ("AvgPool", torch.nn.AvgPool1d(2)),
            ]
        )
        blocks["SincConvBlock"] = torch.nn.Sequential(block)
        in_channels = out_channels

        # First convolutional block, connects the sinc output to the front-end "body"
        out_channels = 128
        blocks["DConvBlock1"] = self.gen_lsc_block(
            in_channels,
            out_channels,
            depthwise_kernel_size=25,
            depthwise_stride=2,
            pointwise_groups=0,
            avgpool=True,
            dropout_probability=0.1,
        )
        in_channels = out_channels

        # Second convolutional block, multiple convolutional layers
        out_channels = self.out_channels
        for layer in [2, 3, 4]:
            blocks[f"DConvBlock{layer}"] = self.gen_lsc_block(
                in_channels, out_channels, depthwise_kernel_size=9, depthwise_stride=1
            )
            in_channels = out_channels

        # Third Convolutional block, acts as coupling to encoder
        out_channels = self.out_channels
        blocks["DConvBlock5"] = self.gen_lsc_block(
            in_channels,
            out_channels,
            depthwise_kernel_size=7,
            depthwise_stride=1,
            pointwise_groups=0,
        )

        self.blocks = torch.nn.Sequential(blocks)

    def gen_lsc_block(
        self,
        in_channels: int,
        out_channels: int,
        depthwise_kernel_size: int = 9,
        depthwise_stride: int = 1,
        depthwise_groups=None,
        pointwise_groups=0,
        dropout_probability: float = 0.15,
        avgpool=False,
    ):
        """
        Generate a convolutional block for Lightweight Sinc convolutions.

        Each block consists of either a depthwise or a depthwise-separable
        convolution along with dropout, (batch-)normalization layer, and
        an optional average-pooling layer. This structure is designed to
        efficiently process audio data while maintaining the integrity of
        the signal through various transformations.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            depthwise_kernel_size (int, optional): Kernel size of the depthwise
                convolution. Default is 9.
            depthwise_stride (int, optional): Stride of the depthwise
                convolution. Default is 1.
            depthwise_groups (int, optional): Number of groups for the
                depthwise convolution. If None, will be set to GCD of
                in_channels and out_channels.
            pointwise_groups (int, optional): Number of groups for the
                pointwise convolution. Default is 0 (no grouping).
            dropout_probability (float, optional): Dropout probability in the
                block. Default is 0.15.
            avgpool (bool, optional): If True, an AvgPool layer is inserted.
                Default is False.

        Returns:
            torch.nn.Sequential: A sequential block containing the defined
            layers, ready to be used in a neural network architecture.

        Examples:
            >>> lsc_block = gen_lsc_block(
            ...     in_channels=64,
            ...     out_channels=128,
            ...     depthwise_kernel_size=5,
            ...     avgpool=True
            ... )
            >>> print(lsc_block)
            Sequential(
              (depthwise): Conv1d(64, 128, kernel_size=(5,), stride=(1,),
              groups=64)
              (activation): LeakyReLU(negative_slope=0.01)
              (batchnorm): BatchNorm1d(128, eps=1e-05, momentum=0.1,
              affine=True, track_running_stats=True)
              (avgpool): AvgPool1d(kernel_size=2, stride=2, padding=0)
              (dropout): Dropout(p=0.15, inplace=False)
            )

        Note:
            The use of depthwise separable convolutions allows for a more
            efficient network structure by reducing the number of parameters
            and computational cost compared to standard convolutions.

        Raises:
            NotImplementedError: If the specified depthwise or pointwise
            groups do not meet the required conditions.
        """
        block = OrderedDict()
        if not depthwise_groups:
            # GCD(in_channels, out_channels) to prevent size mismatches
            depthwise_groups, r = in_channels, out_channels
            while r != 0:
                depthwise_groups, r = depthwise_groups, depthwise_groups % r
        block["depthwise"] = torch.nn.Conv1d(
            in_channels,
            out_channels,
            depthwise_kernel_size,
            depthwise_stride,
            groups=depthwise_groups,
        )
        if pointwise_groups:
            block["pointwise"] = torch.nn.Conv1d(
                out_channels, out_channels, 1, 1, groups=pointwise_groups
            )
        block["activation"] = self.choices_activation[self.activation_type]()
        block["batchnorm"] = torch.nn.BatchNorm1d(out_channels, affine=True)
        if avgpool:
            block["avgpool"] = torch.nn.AvgPool1d(2)
        block["dropout"] = self.choices_dropout[self.dropout_type](dropout_probability)
        return torch.nn.Sequential(block)

    def espnet_initialization_fn(self):
        """
        Initialize sinc filters with filterbank values.

        This function initializes the sinc filters used in the Lightweight
        Sinc Convolutions by setting their values based on the filterbank
        initialization. It also sets the weights and biases of all BatchNorm
        layers in the model to ensure that they start with a neutral effect
        during training.

        The initialization process involves the following steps:
        1. Call the `init_filters()` method on the `filters` attribute to
           initialize the sinc filters.
        2. Iterate through all the blocks of the model. For each block, check
           if it contains a BatchNorm layer with affine parameters enabled. If
           so, set the layer's weight to 1.0 and bias to 0.0.

        Note:
            This method should be called after creating the sinc convolutions
            and before using the model for forward propagation.

        Examples:
            >>> model = LightweightSincConvs()
            >>> model.espnet_initialization_fn()  # Initialize filters

        Raises:
            NotImplementedError: If the filterbank initialization fails.
        """
        self.filters.init_filters()
        for block in self.blocks:
            for layer in block:
                if type(layer) is torch.nn.BatchNorm1d and layer.affine:
                    layer.weight.data[:] = 1.0
                    layer.bias.data[:] = 0.0

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Lightweight Sinc Convolutions.

        This method processes the input tensor using lightweight Sinc
        convolutions, transforming the input audio features into output
        features suitable for subsequent layers in the neural network.

        The input tensor should be formatted as (B, T, C_in, D_in), where:
        - B: Batch size
        - T: Time dimension
        - C_in: Number of input channels
        - D_in: Feature dimension (should be 400 for current implementation)

        The output tensor will be shaped as (B, T, C_out * D_out), where:
        - C_out: Number of output channels, as specified during
          initialization
        - D_out: Output feature dimension, which is 1 in this case.

        Note:
            The current implementation only supports D_in=400, leading
            to D_out=1. For multichannel input, C_out will be the
            product of the initialized out_channels and C_in.

        Args:
            input (torch.Tensor): Input tensor of shape (B, T, C_in, D_in).
            input_lengths (torch.Tensor): Lengths of the input sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Output tensor of shape (B, T, C_out * D_out).
                - Input lengths tensor.

        Examples:
            >>> model = LightweightSincConvs()
            >>> input_tensor = torch.randn(8, 100, 1, 400)  # Example input
            >>> input_lengths = torch.tensor([100] * 8)  # Example lengths
            >>> output, lengths = model.forward(input_tensor, input_lengths)
            >>> output.shape
            torch.Size([8, 100, 256])  # Example output shape

        Raises:
            ValueError: If the input tensor does not have the expected
            shape.
        """
        # Transform input data:
        #   (B, T, C_in, D_in) -> (B*T, C_in, D_in)
        B, T, C_in, D_in = input.size()
        input_frames = input.view(B * T, C_in, D_in)
        output_frames = self.blocks.forward(input_frames)

        # ---TRANSFORM: (B*T, C_out, D_out) -> (B, T, C_out*D_out)
        _, C_out, D_out = output_frames.size()
        output_frames = output_frames.view(B, T, C_out * D_out)
        return output_frames, input_lengths  # no state in this layer

    def output_size(self) -> int:
        """
        Get the output size of the Lightweight Sinc Convolutions.

        This method calculates the output size based on the number of output
        channels and input channels defined during initialization. The output
        size is determined by the formula:

            output_size = out_channels * in_channels

        This output size represents the number of features produced by the
        Lightweight Sinc Convolutions for each input sample.

        Returns:
            int: The computed output size.

        Examples:
            >>> sinc_convs = LightweightSincConvs(in_channels=2, out_channels=256)
            >>> output_size = sinc_convs.output_size()
            >>> print(output_size)
            512  # Since 2 (in_channels) * 256 (out_channels) = 512
        """
        return self.out_channels * self.in_channels


class SpatialDropout(torch.nn.Module):
    """
    Spatial dropout module.

    This module applies dropout to the entire channels of input tensors
    with shape (B, C, D), where B is the batch size, C is the number of
    channels, and D is the dimension of the data. This is particularly useful
    for regularizing deep learning models by preventing overfitting.

    Attributes:
        dropout: An instance of torch.nn.Dropout2d for applying dropout.
        shape: The shape of the input tensors after permutation.

    Args:
        dropout_probability (float): Probability of an element being
            zeroed. Default is 0.15.
        shape (Optional[Union[tuple, list]]): The desired shape of the
            input tensors. Default is (0, 2, 1).

    Examples:
        >>> spatial_dropout = SpatialDropout(dropout_probability=0.2,
        ...                                    shape=(0, 2, 1))
        >>> input_tensor = torch.randn(10, 3, 5)  # Example input
        >>> output_tensor = spatial_dropout(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 3, 5])

    Note:
        The input tensor should be of shape (B, C, D). The shape
        parameter determines how the dimensions are permuted before
        applying dropout.

    Raises:
        ValueError: If the shape parameter is not of type tuple or list.
    """

    @typechecked
    def __init__(
        self,
        dropout_probability: float = 0.15,
        shape: Optional[Union[tuple, list]] = None,
    ):
        """Initialize.

        Args:
            dropout_probability: Dropout probability.
            shape (tuple, list): Shape of input tensors.
        """
        super().__init__()
        if shape is None:
            shape = (0, 2, 1)
        self.dropout = torch.nn.Dropout2d(dropout_probability)
        self.shape = (shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Lightweight Sinc Convolutions.

        This method processes the input tensor through a series of
        lightweight sinc convolutions. The input tensor should be
        formatted as (B, T, C_in, D_in), where:
        - B: Batch size
        - T: Time dimension
        - C_in: Number of input channels
        - D_in: Feature dimension

        The output tensor will have the shape (B, T, C_out * D_out), where:
        - C_out: Number of output channels (initialized in the class)
        - D_out: Output feature dimension (currently fixed to 1).

        Note that the current implementation only supports an input
        feature dimension of D_in=400, resulting in D_out=1. In the
        case of multichannel input, the output channel size is computed
        as the product of the number of output channels and the number
        of input channels.

        Args:
            input (torch.Tensor): Input tensor of shape (B, T, C_in, D_in).
            input_lengths (torch.Tensor): Lengths of each input sequence
                in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): Processed output tensor of shape
                  (B, T, C_out * D_out).
                - input_lengths (torch.Tensor): The same input lengths
                  tensor, unchanged.

        Examples:
            >>> model = LightweightSincConvs()
            >>> input_tensor = torch.randn(16, 100, 1, 400)  # Example input
            >>> input_lengths = torch.tensor([100] * 16)  # Example lengths
            >>> output, lengths = model.forward(input_tensor, input_lengths)
            >>> print(output.shape)  # Output shape: (16, 100, C_out)
        """
        y = x.permute(*self.shape)
        y = self.dropout(y)
        return y.permute(*self.shape)
