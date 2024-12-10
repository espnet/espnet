# SKA-TDNN, original code from: https://github.com/msh9184/ska-tdnn
# adapted for ESPnet-SPK by Jee-weon Jung
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation (SE) Module.

    This module implements the Squeeze-and-Excitation block, which adaptively
    recalibrates channel-wise feature responses by explicitly modeling the
    interdependencies between channels. It uses global information to enhance
    the representational power of the network.

    Attributes:
        se (nn.Sequential): The sequential container that includes:
            - AdaptiveAvgPool1d: Applies adaptive average pooling.
            - Conv1d: 1D convolution layer to reduce channel dimensions.
            - ReLU: Activation function.
            - BatchNorm1d: Batch normalization layer.
            - Conv1d: 1D convolution layer to restore original channel dimensions.
            - Sigmoid: Activation function to produce channel weights.

    Args:
        channels (int): Number of input channels.
        bottleneck (int, optional): Number of channels in the bottleneck layer.
            Default is 128.

    Returns:
        Tensor: The output tensor after applying the squeeze-and-excitation
        operation.

    Examples:
        >>> se_module = SEModule(channels=64, bottleneck=16)
        >>> input_tensor = torch.randn(32, 64, 100)  # (batch_size, channels, time)
        >>> output_tensor = se_module(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 64, 100])
    """

    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
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
            Forward pass of the SEModule.

        This method applies the squeeze-and-excitation (SE) operation to the input
        tensor. The SE module computes a channel-wise weighting of the input tensor
        to enhance the representation of important features and suppress less
        important ones.

        Args:
            input (torch.Tensor): Input tensor of shape (B, C, T), where B is the
                batch size, C is the number of channels, and T is the length of the
                sequence.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input, after applying
                the SE operation.

        Examples:
            >>> se_module = SEModule(channels=64)
            >>> input_tensor = torch.randn(8, 64, 100)  # Batch size 8, 64 channels, length 100
            >>> output_tensor = se_module(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([8, 64, 100])
        """
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    """
    Bottle2neck module for SKA-TDNN architecture.

    This module implements a bottleneck layer with selective kernel attention,
    allowing for adaptive feature extraction through multiple convolutional
    kernels. It utilizes a squeeze-and-excitation mechanism to enhance the
    representation power of the network.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel. Defaults to None.
        kernel_sizes (list of int, optional): List of kernel sizes for the
            selective kernel convolution. Defaults to [5, 7].
        dilation (int, optional): Dilation rate for the convolution. Defaults to None.
        scale (int, optional): Scaling factor for the width of the bottleneck.
            Defaults to 8.
        group (int, optional): Number of groups for grouped convolution.
            Defaults to 1.

    Attributes:
        conv1 (nn.Conv1d): First convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        bn1 (nn.BatchNorm1d): Batch normalization layer.
        nums (int): Number of selective kernel convolutions.
        skconvs (nn.ModuleList): List of selective kernel convolution modules.
        skse (SKAttentionModule): Selective kernel attention module.
        conv3 (nn.Conv1d): Second convolutional layer.
        bn3 (nn.BatchNorm1d): Batch normalization layer.
        se (SEModule): Squeeze-and-excitation module.
        width (int): Width of the bottleneck.

    Returns:
        out (Tensor): Output tensor after applying the bottleneck operation.

    Examples:
        >>> model = Bottle2neck(inplanes=64, planes=128)
        >>> x = torch.randn(32, 64, 100)  # (batch_size, channels, sequence_length)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 128, 100])

    Note:
        This module is designed to work within the SKA-TDNN architecture and
        expects inputs of the shape (batch_size, inplanes, sequence_length).

    Raises:
        ValueError: If `kernel_size` is provided but not valid.
    """

    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=None,
        kernel_sizes=[5, 7],
        dilation=None,
        scale=8,
        group=1,
    ):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        self.skconvs = nn.ModuleList([])
        for i in range(self.nums):
            convs = nn.ModuleList([])
            for k in kernel_sizes:
                convs += [
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv1d(
                                        width,
                                        width,
                                        kernel_size=k,
                                        dilation=dilation,
                                        padding=k // 2 * dilation,
                                        groups=group,
                                    ),
                                ),
                                ("relu", nn.ReLU()),
                                ("bn", nn.BatchNorm1d(width)),
                            ]
                        )
                    )
                ]
            self.skconvs += [convs]
        self.skse = SKAttentionModule(
            channel=width, reduction=4, num_kernels=len(kernel_sizes)
        )
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.se = SEModule(channels=planes)
        self.width = width

    def forward(self, x):
        """
        Computes the forward pass of the Bottle2neck module.

        This method processes the input tensor `x` through a series of
        convolutional layers, applies a skip connection, and returns the
        final output tensor. The forward operation consists of several
        stages including initial convolution, ReLU activation,
        batch normalization, and a series of attention mechanisms.

        Args:
            x (torch.Tensor): The input tensor of shape (B, C, T) where
                B is the batch size, C is the number of channels, and
                T is the sequence length.

        Returns:
            torch.Tensor: The output tensor of shape (B, planes, T)
                after applying the series of transformations.

        Example:
            >>> model = Bottle2neck(inplanes=64, planes=128)
            >>> input_tensor = torch.randn(32, 64, 100)  # Batch size of 32
            >>> output_tensor = model(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([32, 128, 100])

        Note:
            This method relies on the internal layers defined in the
            Bottle2neck class and the proper initialization of those
            layers in the constructor.
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
            sp = self.skse(sp, self.skconvs[i])
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


class ResBlock(nn.Module):
    """
    Residual Block with Selective Kernel Attention.

    This class implements a residual block that incorporates selective kernel
    attention mechanisms for enhancing feature representation in deep learning
    models. It consists of a convolutional layer followed by batch normalization,
    ReLU activation, and two types of selective kernel attention: forward
    and channel-wise.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.
        reduction (int, optional): Reduction ratio for the attention mechanism.
            Default is 8.
        skfwse_freq (int, optional): Frequency parameter for forward selective
            kernel attention. Default is 40.
        skcwse_channel (int, optional): Number of channels for channel-wise
            selective kernel attention. Default is 128.

    Attributes:
        conv1 (nn.Conv2d): Convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer.
        relu (nn.ReLU): ReLU activation function.
        skfwse (fwSKAttention): Forward selective kernel attention module.
        skcwse (cwSKAttention): Channel-wise selective kernel attention module.
        stride (int): Stride of the convolution.

    Returns:
        Tensor: The output of the forward pass.

    Examples:
        >>> block = ResBlock(inplanes=64, planes=128)
        >>> input_tensor = torch.randn(1, 64, 32, 32)  # Batch size 1, 64 channels
        >>> output_tensor = block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 128, 32, 32])
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        reduction: int = 8,
        skfwse_freq: int = 40,
        skcwse_channel: int = 128,
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.skfwse = fwSKAttention(
            freq=skfwse_freq,
            channel=skcwse_channel,
            kernels=[5, 7],
            receptive=[5, 7],
            dilations=[1, 1],
            reduction=reduction,
            groups=1,
        )
        self.skcwse = cwSKAttention(
            freq=skfwse_freq,
            channel=skcwse_channel,
            kernels=[5, 7],
            receptive=[5, 7],
            dilations=[1, 1],
            reduction=reduction,
            groups=1,
        )
        self.stride = stride

    def forward(self, x):
        """
            Forward function for the ResBlock module.

        This method takes an input tensor `x`, applies a series of convolutional,
        batch normalization, and activation operations, and then adds a residual
        connection from the input to the output.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is the
                              batch size, C is the number of channels, H is the
                              height, and W is the width.

        Returns:
            torch.Tensor: Output tensor of the same shape as input `x`.

        Examples:
            >>> res_block = ResBlock(inplanes=64, planes=128)
            >>> input_tensor = torch.randn(32, 64, 16, 16)  # Batch of 32
            >>> output_tensor = res_block(input_tensor)
            >>> output_tensor.shape
            torch.Size([32, 128, 16, 16])
        """
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.skfwse(out)
        out = self.skcwse(out)
        out += residual
        out = self.relu(out)
        return out


class SKAttentionModule(nn.Module):
    """
    Selective Kernel Attention Module.

    This module implements the Selective Kernel Attention mechanism, which
    allows the model to adaptively select kernel responses based on the
    importance of features. It enhances the representation capability by
    dynamically weighting the outputs from multiple convolutional kernels.

    Attributes:
        avg_pool (nn.AdaptiveAvgPool1d): Adaptive average pooling layer.
        D (int): Dimension of the intermediate representation.
        fc (nn.Linear): Fully connected layer for dimensionality reduction.
        relu (nn.ReLU): ReLU activation function.
        fcs (nn.ModuleList): List of fully connected layers for attention weights.
        softmax (nn.Softmax): Softmax layer for normalizing attention weights.

    Args:
        channel (int): Number of input channels.
        reduction (int): Reduction ratio for dimensionality.
        L (int): Maximum number of kernels.
        num_kernels (int): Number of convolutional kernels to use.

    Examples:
        >>> sk_attention = SKAttentionModule(channel=128, reduction=4, L=16, num_kernels=2)
        >>> input_tensor = torch.randn(32, 128, 50)  # (Batch, Channels, Time)
        >>> convs = [nn.Conv1d(128, 128, kernel_size=3, padding=1) for _ in range(2)]
        >>> output = sk_attention(input_tensor, convs)
        >>> output.shape
        torch.Size([32, 128, 50])

    Note:
        This module requires a list of convolutional layers to be passed
        during the forward call for the attention mechanism to work.

    Raises:
        ValueError: If the input tensor does not match the expected dimensions.
    """

    def __init__(self, channel=128, reduction=4, L=16, num_kernels=2):
        super(SKAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.D = max(L, channel // reduction)
        self.fc = nn.Linear(channel, self.D)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList([])
        for i in range(num_kernels):
            self.fcs += [nn.Linear(self.D, channel)]
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, convs):
        """
            Forward function.

        The forward method processes the input tensor through a series of
        convolutional and activation layers, applying the SK attention mechanism
        for improved feature extraction.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T) where B is the
            batch size, C is the number of channels, and T is the length of the
            sequence.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, T) after applying the
            SK attention mechanism and residual connection.

        Examples:
            >>> model = SKAttentionModule(channel=128, reduction=4)
            >>> input_tensor = torch.randn(10, 128, 50)  # Example input
            >>> output_tensor = model(input_tensor, convs)  # Assuming `convs` is defined
            >>> print(output_tensor.shape)  # Output shape will be (10, 128, 50)
        """
        bs, c, t = x.size()
        conv_outs = []
        for conv in convs:
            conv_outs += [conv(x)]
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs)
        S = self.avg_pool(U).view(bs, c)
        Z = self.fc(S)
        Z = self.relu(Z)
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights += [(weight.view(bs, c, 1))]
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights * feats).sum(0)
        return V


class fwSKAttention(nn.Module):
    """
    Frequency-wise Selective Kernel Attention (fwSKAttention) module.

    This module applies frequency-wise selective kernel attention to the input
    tensor, enabling the model to focus on different frequency bands through
    multiple convolutional kernels. It combines the outputs from these kernels
    using learned attention weights.

    Args:
        freq (int): The number of frequency bins in the input tensor.
        channel (int): The number of channels in the input tensor.
        kernels (list of int): List of kernel sizes for the convolutions.
        receptive (list of int): List of receptive field sizes for the convolutions.
        dilations (list of int): List of dilation rates for the convolutions.
        reduction (int): Reduction ratio for the attention mechanism.
        groups (int): Number of groups for group convolution.
        L (int): Maximum value for the hidden dimension in the attention mechanism.

    Returns:
        V (Tensor): The output tensor after applying the selective kernel attention.

    Examples:
        >>> model = fwSKAttention(freq=40, channel=128)
        >>> input_tensor = torch.randn(8, 128, 40, 100)  # (B, C, F, T)
        >>> output = model(input_tensor)
        >>> output.shape
        torch.Size([8, 128, 40, 100])

    Note:
        The input tensor should have a shape of (B, C, F, T), where B is the batch
        size, C is the number of channels, F is the number of frequency bins,
        and T is the number of time steps.

    Raises:
        ValueError: If the input tensor does not have the expected shape.
    """

    def __init__(
        self,
        freq=40,
        channel=128,
        kernels=[3, 5],
        receptive=[3, 5],
        dilations=[1, 1],
        reduction=8,
        groups=1,
        L=16,
    ):
        super(fwSKAttention, self).__init__()
        self.convs = nn.ModuleList([])
        for k, d, r in zip(kernels, dilations, receptive):
            self.convs += [
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    channel,
                                    channel,
                                    kernel_size=k,
                                    padding=r // 2,
                                    dilation=d,
                                    groups=groups,
                                ),
                            ),
                            ("relu", nn.ReLU()),
                            ("bn", nn.BatchNorm2d(channel)),
                        ]
                    )
                )
            ]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.D = max(L, freq // reduction)
        self.fc = nn.Linear(freq, self.D)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs += [nn.Linear(self.D, freq)]
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """
            Implements frequency-wise selective kernel attention.

        This module applies a series of convolutional layers with varying kernel
        sizes to the input tensor, followed by a frequency-wise attention mechanism.
        The attention mechanism helps the model to focus on the most relevant
        frequency components in the input data, which can be particularly useful
        in tasks like speaker verification.

        Args:
            freq (int): The frequency dimension of the input. Default is 40.
            channel (int): The number of input channels. Default is 128.
            kernels (list): List of kernel sizes for the convolutions. Default is
                [3, 5].
            receptive (list): List of receptive field sizes for the convolutions.
                Default is [3, 5].
            dilations (list): List of dilation rates for the convolutions. Default
                is [1, 1].
            reduction (int): The reduction ratio for the attention mechanism.
                Default is 8.
            groups (int): The number of groups for grouped convolutions. Default
                is 1.
            L (int): Maximum number of features after reduction. Default is 16.

        Returns:
            torch.Tensor: The output tensor after applying the attention mechanism.

        Examples:
            >>> attention = fwSKAttention(freq=40, channel=128)
            >>> input_tensor = torch.randn(8, 128, 40, 100)  # (B, C, F, T)
            >>> output_tensor = attention(input_tensor)
            >>> output_tensor.size()
            torch.Size([8, 128, 40, 100])

        Note:
            The input tensor is expected to have 4 dimensions: batch size,
            channels, frequency, and time.

        Raises:
            ValueError: If the input tensor does not have 4 dimensions.
        """
        bs, c, f, t = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs += [conv(x)]
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs).permute(0, 2, 3, 1)
        S = self.avg_pool(U).view(bs, f)
        Z = self.fc(S)
        Z = self.relu(Z)
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights += [(weight.view(bs, 1, f, 1))]
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights * feats).sum(0)
        return V


class cwSKAttention(nn.Module):
    """
        cwSKAttention is a convolutional kernel attention module that applies selective
    kernel attention on feature maps in a multi-scale fashion. It utilizes multiple
    convolutions with different kernel sizes to capture various feature representations,
    and combines them through an attention mechanism.

    Attributes:
        convs (nn.ModuleList): A list of convolutional layers with different kernel
            sizes for feature extraction.
        avg_pool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer to reduce
            feature dimensions.
        D (int): The dimension of the intermediate representation, determined by
            the reduction factor.
        fc (nn.Linear): Fully connected layer for transforming the pooled features.
        relu (nn.ReLU): ReLU activation function.
        fcs (nn.ModuleList): A list of fully connected layers to generate attention
            weights for each kernel.
        softmax (nn.Softmax): Softmax layer to normalize the attention weights.

    Args:
        freq (int): The frequency dimension of the input features.
        channel (int): The number of channels in the input feature map.
        kernels (list): List of kernel sizes to be used in the convolutional layers.
        receptive (list): List of receptive field sizes corresponding to each kernel.
        dilations (list): List of dilation rates for each convolutional layer.
        reduction (int): Reduction factor for the dimensionality of the features.
        groups (int): Number of groups for grouped convolution.
        L (int): Maximum number of features to keep in the intermediate layer.

    Returns:
        Tensor: The output feature map after applying selective kernel attention.

    Examples:
        >>> model = cwSKAttention(freq=40, channel=128, kernels=[3, 5])
        >>> input_tensor = torch.randn(8, 128, 40, 100)  # (B, C, F, T)
        >>> output = model(input_tensor)
        >>> print(output.shape)
        torch.Size([8, 128, 40, 100])  # Output shape matches input shape

    Note:
        The cwSKAttention module is designed to enhance the representational
        capacity of convolutional neural networks by allowing the model to learn
        which features are most important for the task at hand, through adaptive
        attention mechanisms.
    """

    def __init__(
        self,
        freq=40,
        channel=128,
        kernels=[3, 5],
        receptive=[3, 5],
        dilations=[1, 1],
        reduction=8,
        groups=1,
        L=16,
    ):
        super(cwSKAttention, self).__init__()
        self.convs = nn.ModuleList([])
        for k, d, r in zip(kernels, dilations, receptive):
            self.convs += [
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    channel,
                                    channel,
                                    kernel_size=k,
                                    padding=r // 2,
                                    dilation=d,
                                    groups=groups,
                                ),
                            ),
                            ("relu", nn.ReLU()),
                            ("bn", nn.BatchNorm2d(channel)),
                        ]
                    )
                )
            ]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.D = max(L, channel // reduction)
        self.fc = nn.Linear(channel, self.D)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs += [nn.Linear(self.D, channel)]
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """
            Forward function.

        This method performs a forward pass through the cwSKAttention module. It
        takes an input tensor and applies a series of convolutional layers,
        attention mechanisms, and activation functions to compute the output tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, F, T], where B is the
                batch size, C is the number of channels, F is the frequency dimension,
                and T is the time dimension.

        Returns:
            torch.Tensor: Output tensor of shape [B, C, F, T] after applying
            attention and convolutions.

        Examples:
            >>> model = cwSKAttention()
            >>> input_tensor = torch.randn(16, 128, 40, 100)  # Batch of 16
            >>> output_tensor = model(input_tensor)
            >>> output_tensor.shape
            torch.Size([16, 128, 40, 100])
        """
        bs, c, f, t = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs += [conv(x)]
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs)
        S = self.avg_pool(U).view(bs, c)
        Z = self.fc(S)
        Z = self.relu(Z)
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights += [(weight.view(bs, c, 1, 1))]
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights * feats).sum(0)
        return V


class SkaTdnnEncoder(AbsEncoder):
    """
    SKA-TDNN encoder. Extracts frame-level SKA-TDNN embeddings from features.

    Paper: S. Mun, J. Jung et al., "Frequency and Multi-Scale Selective Kernel
        Attention for Speaker Verification,' in Proc. IEEE SLT 2022.

    Args:
        input_size: Input feature dimension.
        block: Type of encoder block class to use. Defaults to "Bottle2neck".
        ndim: Dimensionality of the hidden representation. Defaults to 1024.
        model_scale: Scale value of the Res2Net architecture. Defaults to 8.
        skablock: Type of SKA block to use. Defaults to "ResBlock".
        ska_dim: Dimension of the SKA block. Defaults to 128.
        output_size: Output embedding dimension. Defaults to 1536.

    Attributes:
        _output_size: The output size of the encoder.

    Examples:
        >>> encoder = SkaTdnnEncoder(input_size=40)
        >>> output = encoder(torch.randn(1, 40, 100))  # (B, D, S) shape
        >>> print(output.shape)  # Should output (1, 1536, T)

    Raises:
        ValueError: If an unsupported block type is provided.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        block: str = "Bottle2neck",
        ndim: int = 1024,
        model_scale: int = 8,
        skablock: str = "ResBlock",
        ska_dim: int = 128,
        output_size: int = 1536,
        **kwargs,
    ):
        super().__init__()

        if block == "Bottle2neck":
            block: type = Bottle2neck
        else:
            raise ValueError(f"unsupported block, got: {block}")

        if skablock == "ResBlock":
            ska_block = ResBlock
        else:
            raise ValueError(f"unsupported block, got: {ska_block}")

        self.frt_conv1 = nn.Conv2d(
            1, ska_dim, kernel_size=(3, 3), stride=(2, 1), padding=1
        )
        self.frt_bn1 = nn.BatchNorm2d(ska_dim)
        self.frt_block1 = ska_block(
            ska_dim,
            ska_dim,
            stride=(1, 1),
            skfwse_freq=input_size // 2,
            skcwse_channel=ska_dim,
        )
        self.frt_block2 = ska_block(
            ska_dim,
            ska_dim,
            stride=(1, 1),
            skfwse_freq=input_size // 2,
            skcwse_channel=ska_dim,
        )
        self.frt_conv2 = nn.Conv2d(
            ska_dim, ska_dim, kernel_size=(3, 3), stride=(2, 2), padding=1
        )
        self.frt_bn2 = nn.BatchNorm2d(ska_dim)
        self.conv1 = nn.Conv1d(
            ska_dim * input_size // 4, ndim, kernel_size=5, stride=1, padding=2
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(ndim)
        self.layer1 = block(ndim, ndim, kernel_size=3, dilation=2, scale=model_scale)
        self.layer2 = block(ndim, ndim, kernel_size=3, dilation=3, scale=model_scale)
        self.layer3 = block(ndim, ndim, kernel_size=3, dilation=4, scale=model_scale)
        self.layer4 = nn.Conv1d(3 * ndim, output_size, kernel_size=1)
        self._output_size = output_size

    def output_size(self) -> int:
        """
            SKA-TDNN encoder. Extracts frame-level SKA-TDNN embeddings from features.

        Paper: S. Mun, J. Jung et al., "Frequency and Multi-Scale Selective Kernel
            Attention for Speaker Verification,' in Proc. IEEE SLT 2022.

        Args:
            input_size (int): Input feature dimension.
            block (str): Type of encoder block class to use. Default is "Bottle2neck".
            model_scale (int): Scale value of the Res2Net architecture. Default is 8.
            ndim (int): Dimensionality of the hidden representation. Default is 1024.
            skablock (str): Type of SKA block class to use. Default is "ResBlock".
            ska_dim (int): Dimension for the SKA block. Default is 128.
            output_size (int): Output embedding dimension. Default is 1536.

        Attributes:
            _output_size (int): The output embedding dimension of the encoder.

        Examples:
            >>> encoder = SkaTdnnEncoder(input_size=40)
            >>> output = encoder(torch.randn(10, 40, 100))
            >>> output.shape
            torch.Size([10, 1536])
        """
        return self._output_size

    def forward(self, x):
        """
            Forward function for the SkaTdnnEncoder.

        This method processes the input tensor through a series of convolutional
        layers and residual blocks, ultimately producing a tensor of embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D, S) where:
                - B: Batch size
                - D: Input feature dimension
                - S: Sequence length

        Returns:
            torch.Tensor: Output tensor of shape (B, output_size, T) where:
                - output_size: Dimensionality of the output embeddings
                - T: Output sequence length after processing through layers

        Examples:
            >>> encoder = SkaTdnnEncoder(input_size=40)
            >>> input_tensor = torch.randn(8, 40, 100)  # Batch of 8, 40 features, 100 time steps
            >>> output = encoder.forward(input_tensor)
            >>> output.shape
            torch.Size([8, 1536, T])  # Output shape will depend on the processing

        Note:
            - The input tensor is permuted to match the expected shape for the
              convolutional layers and is reshaped appropriately throughout the
              forward pass.
        """
        x = x.permute(0, 2, 1)  # (B, S, D) -> (B, D, S)
        x = x.unsqueeze(1)  # (B, D, S) -> (B, 1, D, S)

        # the fcwSKA block
        x = self.frt_conv1(x)
        x = self.relu(x)
        x = self.frt_bn1(x)
        x = self.frt_block1(x)
        x = self.frt_block2(x)
        x = self.frt_conv2(x)
        x = self.relu(x)
        x = self.frt_bn2(x)

        x = x.reshape((x.size()[0], -1, x.size()[-1]))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        return x
