import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFMS(nn.Module):
    """
        Alpha-Feature Map Scaling (AFMS) module, which applies scaling to the output of
    each residual block.

    This module is designed to enhance the feature representation in neural networks
    by learning a scaling factor (alpha) for the output features. The scaling factor
    is modulated by a learned feature map that is derived from the input.

    References:
    1. RawNet2: https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    2. AMFS: https://www.koreascience.or.kr/article/JAKO202029757857763.page

    Attributes:
        alpha (nn.Parameter): Learnable scaling factors for each feature dimension.
        fc (nn.Linear): Fully connected layer to project features.
        sig (nn.Sigmoid): Sigmoid activation function.

    Args:
        nb_dim (int): The number of dimensions (features) for the input tensor.

    Returns:
        torch.Tensor: The scaled output tensor after applying the AFMS.

    Examples:
        >>> afms = AFMS(nb_dim=128)
        >>> input_tensor = torch.randn(32, 128, 10)  # (batch_size, nb_dim, seq_len)
        >>> output_tensor = afms(input_tensor)
        >>> print(output_tensor.shape)  # Should be (32, 128, 10)
    """

    def __init__(self, nb_dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((nb_dim, 1)))
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
            Perform the forward pass of the AFMS module.

        This method applies the Alpha-Feature map scaling to the input tensor `x`.
        It computes the adaptive average pooling, applies a fully connected layer,
        and scales the input by the learned alpha parameter.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            torch.Tensor: Output tensor after applying the Alpha-Feature map scaling.

        Examples:
            >>> afms = AFMS(nb_dim=64)
            >>> input_tensor = torch.randn(10, 64, 100)  # (batch_size, channels, length)
            >>> output_tensor = afms(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([10, 64, 100])

        Note:
            This method expects `x` to be a 3-dimensional tensor where the second
            dimension represents the feature channels and the third dimension
            represents the sequence length.

        Raises:
            ValueError: If the input tensor `x` does not have the expected shape.
        """
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)

        x = x + self.alpha
        x = x * y
        return x


class Bottle2neck(nn.Module):
    """
        Bottle2neck is a neural network module that implements a bottleneck architecture
    for processing 1D data, utilizing multiple convolutional layers, batch normalization,
    and residual connections.

    Attributes:
        conv1 (nn.Conv1d): The first convolutional layer.
        bn1 (nn.BatchNorm1d): Batch normalization layer following the first convolution.
        nums (int): The number of intermediate convolutions.
        convs (ModuleList): List of convolutional layers for the bottleneck.
        bns (ModuleList): List of batch normalization layers for the bottleneck.
        conv3 (nn.Conv1d): The final convolutional layer that outputs the processed data.
        bn3 (nn.BatchNorm1d): Batch normalization layer following the final convolution.
        relu (nn.ReLU): ReLU activation function.
        width (int): The width of the intermediate feature maps.
        mp (MaxPool1d or bool): Max pooling layer if pool is specified, otherwise False.
        afms (AFMS): Alpha-Feature map scaling module.
        residual (Sequential or Identity): Residual connection for input data.

    Args:
        inplanes (int): The number of input channels.
        planes (int): The number of output channels.
        kernel_size (int, optional): The size of the convolutional kernel. Defaults to None.
        dilation (int, optional): The dilation rate for convolutions. Defaults to None.
        scale (int, optional): The scaling factor for the number of channels. Defaults to 4.
        pool (bool, optional): Whether to apply max pooling. Defaults to False.

    Returns:
        Tensor: The output tensor after applying the convolutional layers, activation,
        batch normalization, residual connection, and AFMS scaling.

    Examples:
        >>> model = Bottle2neck(inplanes=64, planes=128, kernel_size=3, dilation=1)
        >>> x = torch.randn(10, 64, 100)  # (batch_size, channels, sequence_length)
        >>> output = model(x)
        >>> output.shape
        torch.Size([10, 128, 100])

    Note:
        The architecture allows for flexible scaling of feature maps and incorporates
        residual connections to aid in training deeper networks.

    Todo:
        - Implement additional tests for various configurations of kernel_size and
        dilation.
    """

    def __init__(
        self, inplanes, planes, kernel_size=None, dilation=None, scale=4, pool=False
    ):
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

        self.mp = nn.MaxPool1d(pool) if pool else False
        self.afms = AFMS(planes)

        if inplanes != planes:  # if change in number of filters
            self.residual = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        """
            Forward pass for the Bottle2neck module.

        This method computes the output of the Bottle2neck module given an input tensor.
        It applies several convolutional layers, batch normalization, ReLU activation,
        and performs a residual connection. The output is optionally pooled and passed
        through an Alpha-Feature map scaling (AFMS) module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, inplanes, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, planes, new_length),
            where new_length depends on the pooling layer.

        Examples:
            >>> model = Bottle2neck(inplanes=64, planes=128, kernel_size=3, scale=4)
            >>> input_tensor = torch.randn(8, 64, 100)  # batch_size=8, length=100
            >>> output_tensor = model(input_tensor)
            >>> output_tensor.shape
            torch.Size([8, 128, new_length])  # new_length depends on pooling

        Note:
            If the input and output number of filters are different, a 1x1
            convolution is applied to the input as a residual connection.

        Raises:
            ValueError: If the input tensor dimensions do not match the expected
            shape for the number of input planes.
        """
        residual = self.residual(x)

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

        out += residual
        if self.mp:
            out = self.mp(out)
        out = self.afms(out)

        return out
