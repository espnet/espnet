import math

import torch
import torch.nn as nn

"""
Basic blocks for ECAPA-TDNN.
Code from https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py
"""


class SEModule(nn.Module):
    """
        SEModule implements a Squeeze-and-Excitation (SE) block, which is a crucial
    component of the ECAPA-TDNN architecture. The SE block enhances the representational
    capacity of the network by explicitly modeling the interdependencies between
    channels.

    The SE block performs global average pooling followed by two fully connected
    layers with a ReLU activation in between, and a Sigmoid activation at the end.
    The output of the SE block is multiplied with the input tensor, effectively
    recalibrating the feature maps.

    Attributes:
        channels (int): The number of input channels.
        bottleneck (int): The number of channels in the bottleneck layer. Default is 128.

    Args:
        channels (int): The number of input channels to the SE block.
        bottleneck (int, optional): The number of bottleneck channels. Default is 128.

    Returns:
        Tensor: The output tensor after applying the SE block.

    Examples:
        >>> se_module = SEModule(channels=64, bottleneck=32)
        >>> input_tensor = torch.randn(1, 64, 10)  # (batch_size, channels, sequence_length)
        >>> output_tensor = se_module(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 64, 10])

    Note:
        The SE block is designed to be used as part of larger neural network architectures,
        such as the ECAPA-TDNN, to improve performance on tasks such as speaker
        verification and recognition.
    """

    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        """
                Implements the Squeeze-and-Excitation (SE) module for feature recalibration.

        The SE module applies a squeeze operation followed by an excitation operation
        to recalibrate channel-wise feature responses. This helps in enhancing the
        representative power of the network.

        Attributes:
            se (nn.Sequential): A sequential container that includes:
                - AdaptiveAvgPool1d: A pooling layer that reduces each channel to a
                  single value.
                - Conv1d: A 1D convolution layer to reduce channel dimensions.
                - ReLU: An activation function for introducing non-linearity.
                - BatchNorm1d: A batch normalization layer for stabilizing the learning
                  process.
                - Conv1d: Another 1D convolution layer to restore the original
                  channel dimensions.
                - Sigmoid: An activation function to scale the output between 0 and 1.

        Args:
            channels (int): The number of input channels to the SE module.
            bottleneck (int, optional): The number of bottleneck channels. Default is 128.

        Returns:
            Tensor: The output tensor after applying the SE module to the input.

        Examples:
            >>> import torch
            >>> se_module = SEModule(channels=64)
            >>> input_tensor = torch.rand(1, 64, 10)  # Batch size of 1, 64 channels, 10 length
            >>> output_tensor = se_module(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([1, 64, 10])
        """
        x = self.se(input)
        return input * x


class EcapaBlock(nn.Module):
    """
        Basic blocks for ECAPA-TDNN.
    Code from https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py

    The `EcapaBlock` class implements a building block for the ECAPA-TDNN model,
    which is used in speaker recognition tasks. It incorporates multiple
    convolutional layers, batch normalization, and a squeeze-and-excitation
    module to enhance feature extraction.

    Attributes:
        conv1 (nn.Conv1d): The first 1D convolution layer.
        bn1 (nn.BatchNorm1d): Batch normalization layer following the first conv layer.
        nums (int): Number of convolutional branches (scale - 1).
        convs (ModuleList): List of convolutional layers for each branch.
        bns (ModuleList): List of batch normalization layers for each branch.
        conv3 (nn.Conv1d): The final 1D convolution layer.
        bn3 (nn.BatchNorm1d): Batch normalization layer following the final conv layer.
        relu (ReLU): ReLU activation function.
        width (int): Width of each convolutional branch.
        se (SEModule): Squeeze-and-excitation module for channel-wise attention.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to None.
        dilation (int, optional): Dilation rate for convolution. Defaults to None.
        scale (int, optional): Scale factor for width. Defaults to 8.

    Returns:
        Tensor: The output tensor after applying the ECAPA block.

    Examples:
        >>> ecapa_block = EcapaBlock(inplanes=64, planes=128, kernel_size=3)
        >>> input_tensor = torch.randn(1, 64, 100)  # Batch size of 1, 64 channels, 100 length
        >>> output_tensor = ecapa_block(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 128, 100])  # Output should have 128 channels
    """

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        """
                Forward pass of the ECAPA block.

        This method processes the input tensor through a series of convolutional
        layers, batch normalization, and activation functions. It also includes
        a squeeze-and-excitation module to enhance the feature representation.
        The output is the result of adding the residual connection to the processed
        features.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels,
                              sequence_length).

        Returns:
            torch.Tensor: The output tensor after applying the ECAPA block, with
                          shape (batch_size, out_channels, sequence_length).

        Examples:
            >>> ecapa_block = EcapaBlock(inplanes=64, planes=128, kernel_size=3,
                                          dilation=2)
            >>> input_tensor = torch.randn(8, 64, 100)  # batch_size=8, channels=64, seq_len=100
            >>> output_tensor = ecapa_block(input_tensor)
            >>> output_tensor.shape
            torch.Size([8, 128, 100])
        """
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out
