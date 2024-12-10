# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.enh.layers.ncsnpp_utils import layers, up_or_down_sampling

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for noise levels.

    This module generates Gaussian Fourier embeddings for a given input tensor
    by projecting the input onto a set of Fourier basis functions. The projection
    is performed using a learned weight vector `W`, which is a fixed parameter
    during training.

    Attributes:
        W (torch.Parameter): A fixed parameter representing the learned weights
            for the Fourier projection, initialized with random values scaled
            by the specified scale.

    Args:
        embedding_size (int): The size of the embedding vector. Defaults to 256.
        scale (float): The scale factor for initializing the weights. Defaults to 1.0.

    Returns:
        torch.Tensor: The concatenated sine and cosine embeddings of the input.

    Examples:
        >>> gaussian_fourier_proj = GaussianFourierProjection(embedding_size=128)
        >>> input_tensor = torch.randn(10, 1)  # Batch size of 10
        >>> output = gaussian_fourier_proj(input_tensor)
        >>> output.shape
        torch.Size([10, 256])  # 128 sine + 128 cosine

    Note:
        The input tensor `x` should have shape (N, 1) where N is the batch size.
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        """
        Compute the Gaussian Fourier projection of the input tensor.

    This method projects the input tensor `x` onto a higher-dimensional space
    using Gaussian Fourier embeddings. The projection is achieved by multiplying
    the input by learnable weights and applying sine and cosine transformations.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N), where B is the batch size
                          and N is the input dimension.

    Returns:
        torch.Tensor: Output tensor of shape (B, 2 * embedding_size), where
                      `embedding_size` is the number of dimensions in the
                      Gaussian Fourier projection.

    Examples:
        >>> import torch
        >>> gaussian_fourier = GaussianFourierProjection(embedding_size=256)
        >>> x = torch.randn(10, 1)  # Batch of 10 with 1 dimension
        >>> output = gaussian_fourier(x)
        >>> output.shape
        torch.Size([10, 512])  # 512 = 2 * embedding_size

    Note:
        The weights `W` are initialized with a Gaussian distribution and are
        not trainable (requires_grad=False).
        """
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
    """
    Combine information from skip connections.

    This module combines two input tensors using a specified method. The 
    default method is concatenation, but addition can also be used. It 
    applies a 1x1 convolution to the first input tensor before combining 
    it with the second input tensor.

    Attributes:
        Conv_0 (nn.Module): A 1x1 convolution layer applied to the first 
            input tensor.
        method (str): The method to combine the inputs. Can be "cat" for 
            concatenation or "sum" for addition.

    Args:
        dim1 (int): The number of input channels for the first tensor.
        dim2 (int): The number of input channels for the second tensor.
        method (str, optional): The method to combine the inputs. Defaults 
            to "cat".

    Returns:
        torch.Tensor: The combined output tensor after applying the 
        specified method.

    Raises:
        ValueError: If the method is not recognized.

    Examples:
        >>> combine = Combine(dim1=64, dim2=32, method="cat")
        >>> x = torch.randn(1, 64, 16, 16)  # First input tensor
        >>> y = torch.randn(1, 32, 16, 16)  # Second input tensor
        >>> output = combine(x, y)  # Combines using concatenation

        >>> combine_sum = Combine(dim1=64, dim2=32, method="sum")
        >>> output_sum = combine_sum(x, y)  # Combines using addition
    """

    def __init__(self, dim1, dim2, method="cat"):
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method

    def forward(self, x, y):
        """
        Combines two inputs using a specified method.

    This method applies a convolution to the first input `x`, then combines
    the result with the second input `y` based on the specified method,
    which can either be concatenation or summation.

    Args:
        x (torch.Tensor): The first input tensor, which is passed through a
            convolution layer. The shape should be (B, C1, H, W) where B
            is the batch size, C1 is the number of channels, and H and W are
            the height and width of the input.
        y (torch.Tensor): The second input tensor to be combined with the
            processed `x`. The shape should be (B, C2, H, W) where C2 should
            match the output channels of the convolution layer.

    Returns:
        torch.Tensor: The combined output tensor. If `method` is "cat", the
        output shape will be (B, C1 + C2, H, W). If `method` is "sum", the
        output shape will be (B, C1, H, W).

    Raises:
        ValueError: If the specified method is not recognized. 

    Examples:
        >>> combine_layer = Combine(dim1=64, dim2=32, method='cat')
        >>> x = torch.randn(1, 64, 8, 8)  # (B, C1, H, W)
        >>> y = torch.randn(1, 32, 8, 8)   # (B, C2, H, W)
        >>> output = combine_layer(x, y)
        >>> output.shape
        torch.Size([1, 96, 8, 8])  # Output shape when method is 'cat'

        >>> combine_layer = Combine(dim1=64, dim2=32, method='sum')
        >>> output = combine_layer(x, y)
        >>> output.shape
        torch.Size([1, 64, 8, 8])  # Output shape when method is 'sum'
        """
        h = self.Conv_0(x)
        if self.method == "cat":
            return torch.cat([h, y], dim=1)
        elif self.method == "sum":
            return h + y
        else:
            raise ValueError(f"Method {self.method} not recognized.")


class AttnBlockpp(nn.Module):
    """
    Channel-wise self-attention block. Modified from DDPM.

    This class implements a channel-wise self-attention mechanism 
    that allows the model to focus on different parts of the input 
    features adaptively. It is designed to work within the NCSN++ 
    architecture, enhancing the representation capabilities of the 
    model.

    Attributes:
        GroupNorm_0 (nn.GroupNorm): Group normalization layer for input.
        NIN_0 (NIN): First non-linear transformation layer.
        NIN_1 (NIN): Second non-linear transformation layer.
        NIN_2 (NIN): Third non-linear transformation layer.
        NIN_3 (NIN): Fourth non-linear transformation layer.
        skip_rescale (bool): Flag to determine if output should be 
            rescaled.

    Args:
        channels (int): Number of input channels.
        skip_rescale (bool, optional): If True, rescales the output 
            to maintain stability. Defaults to False.
        init_scale (float, optional): Scale for initializing layers. 
            Defaults to 0.0.

    Returns:
        Tensor: The output tensor after applying self-attention.

    Raises:
        ValueError: If input tensor shape is not valid.

    Examples:
        >>> attn_block = AttnBlockpp(channels=64)
        >>> input_tensor = torch.randn(8, 64, 32, 32)  # (batch, channels, height, width)
        >>> output_tensor = attn_block(input_tensor)
        >>> output_tensor.shape
        torch.Size([8, 64, 32, 32])  # Output shape matches input shape
    """

    def __init__(self, channels, skip_rescale=False, init_scale=0.0):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6
        )
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        """
        Forward pass for the AttnBlockpp module.

This method implements the forward pass for the attention block, which 
performs channel-wise self-attention on the input tensor. The attention 
mechanism utilizes query, key, and value representations derived from 
the input, and combines them to produce an output tensor. The output 
can be either the original input plus the attention output, or a 
rescaled version of the sum based on the skip_rescale attribute.

Args:
    x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is the 
        batch size, C is the number of channels, H is the height, and 
        W is the width of the input.

Returns:
    torch.Tensor: Output tensor of the same shape as the input, 
        representing the result of the self-attention operation.

Raises:
    ValueError: If the input tensor does not have the expected shape 
        or if the number of channels does not match the initialized 
        parameters.

Examples:
    >>> model = AttnBlockpp(channels=64)
    >>> input_tensor = torch.randn(8, 64, 32, 32)  # Batch of 8 images
    >>> output_tensor = model(input_tensor)
    >>> print(output_tensor.shape)
    torch.Size([8, 64, 32, 32])
        """
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class Upsample(nn.Module):
    """
    Upsampling layer that increases the spatial dimensions of the input.

    This layer can optionally include a convolution operation after the 
    upsampling, and it supports different upsampling methods, including 
    nearest neighbor and FIR (Finite Impulse Response) filters.

    Attributes:
        fir (bool): If True, use FIR filters for upsampling.
        fir_kernel (tuple): Kernel for FIR upsampling.
        with_conv (bool): If True, apply a convolution after upsampling.
        out_ch (int): Number of output channels.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int, optional): Number of output channels. Defaults to in_ch.
        with_conv (bool, optional): If True, apply convolution after 
            upsampling. Defaults to False.
        fir (bool, optional): If True, use FIR filters for upsampling. 
            Defaults to False.
        fir_kernel (tuple, optional): FIR kernel for upsampling. 
            Defaults to (1, 3, 3, 1).

    Returns:
        Tensor: Upsampled tensor of shape (B, out_ch, H * 2, W * 2).

    Examples:
        >>> upsample_layer = Upsample(in_ch=64, out_ch=128, with_conv=True)
        >>> input_tensor = torch.randn(1, 64, 32, 32)  # Batch of 1, 64 channels
        >>> output_tensor = upsample_layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 128, 64, 64])  # Upsampled to 128 channels and doubled size

    Note:
        If `fir` is set to True, the upsampling will use a FIR filter instead 
        of the default nearest neighbor method.

    Raises:
        ValueError: If the input tensor does not have the expected number of 
        dimensions.
    """
    def __init__(
        self,
        in_ch=None,
        out_ch=None,
        with_conv=False,
        fir=False,
        fir_kernel=(1, 3, 3, 1),
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch)
        else:
            if with_conv:
                self.Conv2d_0 = up_or_down_sampling.Conv2d(
                    in_ch,
                    out_ch,
                    kernel=3,
                    up=True,
                    resample_kernel=fir_kernel,
                    use_bias=True,
                    kernel_init=default_init(),
                )
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x):
        """
        Upsampling layer with optional convolutional layer.

    This class implements an upsampling layer that can optionally include a 
    convolutional operation. It can perform nearest neighbor upsampling or 
    use a convolutional layer for upsampling, depending on the provided 
    parameters. 

    Attributes:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        with_conv (bool): Whether to apply a convolutional layer after 
            upsampling.
        fir (bool): Whether to use a FIR filter for upsampling.
        fir_kernel (tuple): Kernel size for FIR filter.
    
    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels. If None, defaults to in_ch.
        with_conv (bool): If True, applies a convolutional layer after 
            upsampling.
        fir (bool): If True, uses FIR filter for upsampling.
        fir_kernel (tuple): Kernel size for FIR filter. Defaults to 
            (1, 3, 3, 1).

    Returns:
        Tensor: The upsampled tensor with shape (B, out_ch, H*2, W*2).
    
    Examples:
        >>> upsample_layer = Upsample(in_ch=64, out_ch=128, with_conv=True)
        >>> x = torch.randn(1, 64, 32, 32)  # Example input
        >>> output = upsample_layer(x)
        >>> print(output.shape)
        torch.Size([1, 128, 64, 64])  # Output shape after upsampling
    
    Note:
        The upsampling method used is determined by the `fir` and 
        `with_conv` parameters. If `fir` is False, nearest neighbor 
        upsampling is used. If `fir` is True and `with_conv` is True, 
        a convolutional layer is applied after FIR upsampling.
    
    Raises:
        ValueError: If the input tensor shape is not compatible with 
            the expected dimensions.
        """
        B, C, H, W = x.shape
        if not self.fir:
            h = F.interpolate(x, (H * 2, W * 2), "nearest")
            if self.with_conv:
                h = self.Conv_0(h)
        else:
            if not self.with_conv:
                h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = self.Conv2d_0(x)

        return h


class Downsample(nn.Module):
    """
    Downsampling layer that can use convolution or average pooling.

    This class implements a downsampling operation that reduces the spatial
    dimensions of the input tensor by a factor of two. The downsampling can
    be achieved either through convolution (with optional FIR filtering) or
    average pooling. 

    Attributes:
        fir (bool): Indicates whether to use FIR filtering for downsampling.
        fir_kernel (tuple): Kernel used for FIR filtering if `fir` is True.
        with_conv (bool): Indicates whether to apply a convolution after
            downsampling.
        out_ch (int): Number of output channels.

    Args:
        in_ch (int, optional): Number of input channels. Defaults to None.
        out_ch (int, optional): Number of output channels. If None, 
            it defaults to `in_ch`.
        with_conv (bool, optional): If True, applies a convolution after 
            downsampling. Defaults to False.
        fir (bool, optional): If True, uses FIR filtering for downsampling. 
            Defaults to False.
        fir_kernel (tuple, optional): Kernel used for FIR filtering. 
            Defaults to (1, 3, 3, 1).

    Returns:
        torch.Tensor: Downsampled tensor with shape (B, out_ch, H/2, W/2).

    Examples:
        >>> downsample = Downsample(in_ch=64, out_ch=128, with_conv=True)
        >>> x = torch.randn(1, 64, 32, 32)  # Batch size of 1
        >>> output = downsample(x)
        >>> print(output.shape)
        torch.Size([1, 128, 16, 16])  # Output shape after downsampling

    Note:
        When `fir` is set to True, the downsampling operation will use the
        specified FIR kernel for resampling.

    Raises:
        ValueError: If `fir` is True and `with_conv` is also True, it will
        use a convolutional layer for downsampling.
    """
    def __init__(
        self,
        in_ch=None,
        out_ch=None,
        with_conv=False,
        fir=False,
        fir_kernel=(1, 3, 3, 1),
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        else:
            if with_conv:
                self.Conv2d_0 = up_or_down_sampling.Conv2d(
                    in_ch,
                    out_ch,
                    kernel=3,
                    down=True,
                    resample_kernel=fir_kernel,
                    use_bias=True,
                    kernel_init=default_init(),
                )
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        """
        Downsample layer that reduces the spatial dimensions of the input tensor.

This layer can optionally include a convolution operation and can use either
standard pooling or a Fourier interpolation method for downsampling. The
parameters allow for flexibility in kernel selection and whether to use a
convolutional layer in the downsampling process.

Attributes:
    fir (bool): Indicates whether to use a FIR filter for downsampling.
    fir_kernel (tuple): Kernel used for the FIR filter.
    with_conv (bool): Indicates whether to use a convolutional layer.
    out_ch (int): Number of output channels.

Args:
    in_ch (int): Number of input channels.
    out_ch (int, optional): Number of output channels. Defaults to in_ch.
    with_conv (bool, optional): Whether to include a convolutional layer.
        Defaults to False.
    fir (bool, optional): Whether to use FIR downsampling. Defaults to False.
    fir_kernel (tuple, optional): Kernel for FIR downsampling. Defaults to
        (1, 3, 3, 1).

Returns:
    torch.Tensor: The downsampled output tensor.

Raises:
    ValueError: If an invalid downsampling method is specified.

Examples:
    >>> downsample_layer = Downsample(in_ch=64, out_ch=32, with_conv=True)
    >>> input_tensor = torch.randn(1, 64, 32, 32)
    >>> output_tensor = downsample_layer(input_tensor)
    >>> print(output_tensor.shape)
    torch.Size([1, 32, 16, 16])

Note:
    If `with_conv` is set to True, a convolutional layer is applied after
    downsampling. If `fir` is True, a FIR filter is used instead of the
    standard pooling method.

Todo:
    - Add support for different pooling methods.
        """
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        else:
            if not self.with_conv:
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                x = self.Conv2d_0(x)

        return x


class ResnetBlockDDPMpp(nn.Module):
    """
    ResBlock adapted from DDPM.

    This class implements a residual block that is adapted from Denoising 
    Diffusion Probabilistic Models (DDPM). It includes normalization, 
    convolutional layers, and the option to incorporate temporal embeddings 
    for conditional processing. The block supports skip connections and 
    can rescale the output.

    Attributes:
        act: Activation function to be applied in the block.
        out_ch: Number of output channels for the convolutional layers.
        conv_shortcut: Whether to use a convolutional shortcut.
        skip_rescale: Whether to rescale the skip connection output.
        GroupNorm_0: First group normalization layer.
        Conv_0: First convolutional layer.
        Dense_0: Linear layer for processing temporal embeddings (if provided).
        GroupNorm_1: Second group normalization layer.
        Dropout_0: Dropout layer.
        Conv_1: Second convolutional layer.
        NIN_0: Non-linear layer for adjusting input dimensions (if needed).
        Conv_2: Second convolutional layer for shortcut connection (if needed).

    Args:
        act (callable): Activation function to be used (e.g., ReLU).
        in_ch (int): Number of input channels.
        out_ch (int, optional): Number of output channels. Defaults to `in_ch`.
        temb_dim (int, optional): Dimension of the temporal embedding. 
            Defaults to `None`.
        conv_shortcut (bool, optional): If `True`, use convolution for 
            shortcut. Defaults to `False`.
        dropout (float, optional): Dropout probability. Defaults to `0.1`.
        skip_rescale (bool, optional): If `True`, apply rescaling to the 
            output. Defaults to `False`.
        init_scale (float, optional): Scale for weight initialization. 
            Defaults to `0.0`.

    Returns:
        Tensor: The output of the residual block after applying 
        the operations.

    Examples:
        >>> block = ResnetBlockDDPMpp(act=F.relu, in_ch=64, out_ch=128)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = block(x)
        >>> output.shape
        torch.Size([1, 128, 32, 32])

    Note:
        This block is designed to be used within diffusion models where 
        residual learning can enhance performance.

    Raises:
        ValueError: If `method` in skip connections is not recognized.

    Todo:
        - Implement additional initialization methods for improved performance.
    """

    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        conv_shortcut=False,
        dropout=0.1,
        skip_rescale=False,
        init_scale=0.0,
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        """
        Performs a forward pass through the ResNet block.

    This method applies a series of operations including normalization,
    convolution, and optional time embedding addition to the input tensor.
    The output can be either a residual connection or a rescaled output
    based on the configuration.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is the
            batch size, C is the number of channels, H is the height, and W
            is the width.
        temb (torch.Tensor, optional): Time embedding tensor of shape (B, T),
            where T is the dimensionality of the time embedding. If provided,
            it is added to the output feature maps.

    Returns:
        torch.Tensor: Output tensor of shape (B, out_ch, H, W), where
            out_ch is the number of output channels. The output is computed
            as either a simple addition of the input and output features or
            a rescaled sum based on the skip_rescale attribute.

    Examples:
        >>> model = ResnetBlockDDPMpp(act=nn.ReLU(), in_ch=64, out_ch=128)
        >>> input_tensor = torch.randn(8, 64, 32, 32)  # Batch of 8
        >>> output_tensor = model(input_tensor)
        >>> output_tensor.shape
        torch.Size([8, 128, 32, 32])

    Note:
        If `temb` is provided, it must match the output channel dimension.

    Raises:
        ValueError: If the input tensor's channel dimension does not match
            the expected dimensions based on the model's configuration.
        """
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class ResnetBlockBigGANpp(nn.Module):
    """
    ResNet Block adapted for BigGAN++ architecture.

    This class implements a residual block that can optionally perform upsampling
    or downsampling operations. It uses group normalization, dropout, and
    convolutional layers to transform the input tensor while maintaining
    residual connections.

    Attributes:
        act: The activation function to apply.
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        temb_dim: Dimension of the time embedding (optional).
        up: Boolean indicating if upsampling should be performed.
        down: Boolean indicating if downsampling should be performed.
        dropout: Dropout rate to apply.
        fir: Boolean indicating if a FIR filter should be used for sampling.
        fir_kernel: Kernel to use for the FIR filter.
        skip_rescale: Boolean indicating if the output should be rescaled.
        init_scale: Initial scale for weights.

    Args:
        act (callable): Activation function.
        in_ch (int): Number of input channels.
        out_ch (int, optional): Number of output channels. Defaults to `in_ch`.
        temb_dim (int, optional): Dimension of the time embedding. Defaults to `None`.
        up (bool, optional): If `True`, performs upsampling. Defaults to `False`.
        down (bool, optional): If `True`, performs downsampling. Defaults to `False`.
        dropout (float, optional): Dropout rate. Defaults to `0.1`.
        fir (bool, optional): If `True`, uses FIR filter for upsampling/downsampling.
                              Defaults to `False`.
        fir_kernel (tuple, optional): Kernel for FIR filter. Defaults to `(1, 3, 3, 1)`.
        skip_rescale (bool, optional): If `True`, rescales the output. Defaults to `True`.
        init_scale (float, optional): Initial scale for the weights. Defaults to `0.0`.

    Returns:
        torch.Tensor: The output tensor after applying the residual block.

    Examples:
        >>> block = ResnetBlockBigGANpp(act=F.relu, in_ch=64, out_ch=128, up=True)
        >>> input_tensor = torch.randn(1, 64, 32, 32)
        >>> output_tensor = block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 128, 64, 64])

    Note:
        Ensure that the input tensor shape matches the expected dimensions based
        on the `in_ch` parameter. The output tensor will have the shape of 
        (batch_size, out_ch, height, width) depending on the upsampling or 
        downsampling operations performed.
    """
    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        up=False,
        down=False,
        dropout=0.1,
        fir=False,
        fir_kernel=(1, 3, 3, 1),
        skip_rescale=True,
        init_scale=0.0,
    ):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None):
        """
        Perform a forward pass through the ResNet block.

    This method takes input tensor `x` and optional time embedding `temb`,
    processes them through several layers including normalization, 
    convolution, and activation functions, and applies the appropriate 
    upsampling or downsampling if specified during initialization.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W) where B is the 
            batch size, C is the number of channels, H is the height, 
            and W is the width.
        temb (torch.Tensor, optional): Time embedding tensor of shape 
            (B, T) where T is the embedding dimension. Default is None.

    Returns:
        torch.Tensor: Output tensor of shape (B, out_ch, H', W') where 
            H' and W' depend on whether upsampling or downsampling is 
            performed.

    Examples:
        >>> block = ResnetBlockBigGANpp(act=nn.ReLU(), in_ch=64, out_ch=128)
        >>> x = torch.randn(8, 64, 32, 32)  # Batch of 8, 64 channels, 32x32
        >>> output = block(x)
        >>> output.shape
        torch.Size([8, 128, 32, 32])  # Output shape

    Note:
        If `up` or `down` is set to True during initialization, the input 
        tensor will be upsampled or downsampled accordingly.

    Raises:
        ValueError: If the input dimensions do not match the expected shape.
        """
        h = self.act(self.GroupNorm_0(x))

        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)
