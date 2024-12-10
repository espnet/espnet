"""Layers used for up-sampling or down-sampling images.

Many functions are ported from https://github.com/NVlabs/stylegan2.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.enh.layers.ncsnpp_utils.upfirdn2d import upfirdn2d


# Function ported from StyleGAN2
def get_weight(module, shape, weight_var="weight", kernel_init=None):
    """
    Get/create weight tensor for a convolution or fully-connected layer.

    This function retrieves or initializes the weight tensor for a specified
    layer within a neural network module. It allows for the creation of
    weights with a given shape and optional kernel initialization.

    Args:
        module (nn.Module): The neural network module from which to retrieve or
            create the weight tensor.
        shape (tuple): The desired shape of the weight tensor.
        weight_var (str, optional): The variable name of the weight tensor
            (default is "weight").
        kernel_init (callable, optional): A function to initialize the weight
            tensor. If None, the weights will be uninitialized.

    Returns:
        torch.Tensor: The initialized or retrieved weight tensor.

    Examples:
        >>> import torch
        >>> module = nn.Linear(10, 5)
        >>> weight = get_weight(module, (5, 10))
        >>> weight.shape
        torch.Size([5, 10])

        >>> def custom_init(shape):
        ...     return torch.randn(shape) * 0.01
        >>> weight = get_weight(module, (5, 10), kernel_init=custom_init)
        >>> weight.mean()
        tensor(..., grad_fn=<MeanBackward0>)

    Note:
        The function assumes that the module passed in has the capability to
        store a parameter with the specified weight variable name.
    """

    return module.param(weight_var, kernel_init, shape)


class Conv2d(nn.Module):
    """
        Conv2d layer with optimal upsampling and downsampling (StyleGAN2).

    This class implements a 2D convolutional layer that supports both upsampling
    and downsampling operations, inspired by the methods used in StyleGAN2. The
    layer is designed to be efficient and flexible, allowing for the application
    of various kernel sizes and resampling techniques.

    Attributes:
        weight (torch.nn.Parameter): The learnable weight parameter for the
            convolution operation.
        bias (torch.nn.Parameter, optional): The learnable bias parameter for the
            convolution operation, if use_bias is True.
        up (bool): Indicates whether the layer performs upsampling.
        down (bool): Indicates whether the layer performs downsampling.
        resample_kernel (tuple): The kernel used for resampling during
            upsampling/downsampling.
        kernel (int): The size of the convolution kernel.
        use_bias (bool): Flag to indicate if bias should be used.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel (int): Size of the convolution kernel (must be odd and >= 1).
        up (bool, optional): If True, the layer will perform upsampling.
            Default is False.
        down (bool, optional): If True, the layer will perform downsampling.
            Default is False.
        resample_kernel (tuple, optional): Kernel for resampling. Default is
            (1, 3, 3, 1).
        use_bias (bool, optional): If True, a bias term will be added to the
            output. Default is True.
        kernel_init (callable, optional): Function to initialize the kernel weights.

    Returns:
        torch.Tensor: The output tensor after applying the convolution,
        upsampling, or downsampling.

    Raises:
        AssertionError: If both up and down are set to True, or if kernel size
            is not valid.

    Examples:
        >>> conv_layer = Conv2d(in_ch=3, out_ch=64, kernel=3, up=True)
        >>> input_tensor = torch.randn(1, 3, 64, 64)
        >>> output_tensor = conv_layer(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 64, 128, 128])

        >>> conv_layer = Conv2d(in_ch=3, out_ch=64, kernel=3, down=True)
        >>> output_tensor = conv_layer(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 64, 32, 32])
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel,
        up=False,
        down=False,
        resample_kernel=(1, 3, 3, 1),
        use_bias=True,
        kernel_init=None,
    ):
        super().__init__()
        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))

        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias

    def forward(self, x):
        """
            Conv2d layer with optimal upsampling and downsampling (StyleGAN2).

        This class implements a convolutional layer that can perform
        upsampling, downsampling, or standard convolution based on the
        parameters provided. It uses optimized techniques inspired by
        the StyleGAN2 architecture for improved performance.

        Attributes:
            weight (nn.Parameter): The learnable weight tensor for the convolution.
            bias (nn.Parameter, optional): The learnable bias tensor for the convolution.
            up (bool): Flag to indicate if upsampling should be performed.
            down (bool): Flag to indicate if downsampling should be performed.
            resample_kernel (tuple): The kernel used for resampling.
            kernel (int): The size of the convolution kernel.
            use_bias (bool): Flag to indicate if bias should be used in the convolution.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            kernel (int): Size of the convolution kernel (must be odd).
            up (bool, optional): If True, the layer will perform upsampling. Defaults to False.
            down (bool, optional): If True, the layer will perform downsampling. Defaults to False.
            resample_kernel (tuple, optional): FIR filter kernel for resampling. Defaults to (1, 3, 3, 1).
            use_bias (bool, optional): If True, adds a bias term. Defaults to True.
            kernel_init (callable, optional): A function to initialize the kernel weights. Defaults to None.

        Raises:
            AssertionError: If both `up` and `down` are set to True or if the `kernel` size is not valid.

        Examples:
            >>> conv_layer = Conv2d(in_ch=3, out_ch=16, kernel=3, up=True)
            >>> input_tensor = torch.randn(1, 3, 64, 64)  # Batch of 3-channel images
            >>> output_tensor = conv_layer(input_tensor)
            >>> print(output_tensor.shape)  # Output shape after upsampling

        Note:
            This implementation is inspired by the StyleGAN2 architecture,
            and aims to provide efficient upsampling and downsampling operations.
        """
        if self.up:
            x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
        else:
            x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

        if self.use_bias:
            x = x + self.bias.reshape(1, -1, 1, 1)

        return x


def naive_upsample_2d(x, factor=2):
    """
    Perform naive 2D upsampling on a tensor.

    This function takes an input tensor representing a batch of 2D images and
    upsamples each image by the specified factor using nearest-neighbor
    interpolation. The output tensor will have its height and width multiplied
    by the upsampling factor.

    Args:
        x (torch.Tensor): Input tensor of shape `[N, C, H, W]`, where:
            - N: Number of images in the batch.
            - C: Number of channels in each image.
            - H: Height of each image.
            - W: Width of each image.
        factor (int): Upsampling factor. Default is 2.

    Returns:
        torch.Tensor: Output tensor of shape `[N, C, H * factor, W * factor]`.

    Examples:
        >>> x = torch.rand(1, 3, 4, 4)  # A batch of 1 image with 3 channels
        >>> upsampled = naive_upsample_2d(x, factor=2)
        >>> print(upsampled.shape)
        torch.Size([1, 3, 8, 8])  # The output shape after upsampling

    Note:
        This method is a simple and efficient way to increase the resolution of
        images but may not preserve the quality as well as more advanced methods.
    """
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
    """
    Downsample a 4D tensor (batch of 2D images) by averaging.

    This function takes a batch of 2D images and downsamples each image by
    a specified factor using a naive averaging method. The input tensor
    should have the shape `[N, C, H, W]`, where `N` is the batch size,
    `C` is the number of channels, `H` is the height, and `W` is the width.
    The output tensor will have the shape `[N, C, H // factor, W // factor]`.

    Args:
        x (torch.Tensor): Input tensor of shape `[N, C, H, W]` to be
            downsampled.
        factor (int, optional): The downsampling factor. Must be a positive
            integer (default is 2).

    Returns:
        torch.Tensor: A downsampled tensor of shape `[N, C, H // factor,
        W // factor]`.

    Examples:
        >>> import torch
        >>> x = torch.rand(1, 3, 4, 4)  # A single 4x4 image with 3 channels
        >>> downsampled_x = naive_downsample_2d(x, factor=2)
        >>> print(downsampled_x.shape)  # Output will be [1, 3, 2, 2]
    """
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5))


def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
    """
        Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

    Padding is performed only once at the beginning, not between the
    operations. The fused operation is considerably more efficient than
    performing the same calculation using standard TensorFlow ops. It
    supports gradients of arbitrary order.

    Args:
        x: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        w: Weight tensor of the shape `[filterH, filterW, inChannels,
           outChannels]`. Grouped convolution can be performed by
           `inChannels = x.shape[0] // numGroups`.
        k: FIR filter of the shape `[firH, firW]` or `[firN]`
           (separable). The default is `[1] * factor`, which corresponds
           to nearest-neighbor upsampling.
        factor: Integer upsampling factor (default: 2).
        gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]` or
        `[N, H * factor, W * factor, C]`, and same datatype as `x`.

    Examples:
        >>> x = torch.randn(1, 3, 4, 4)  # A batch of 1 image with 3 channels
        >>> w = torch.randn(3, 3, 3, 3)  # Example weight tensor
        >>> output = upsample_conv_2d(x, w, factor=2)  # Upsampling by a factor of 2

    Note:
        Ensure that the input tensor `x` and weight tensor `w` are
        correctly shaped for the operation to work without errors.

    Raises:
        AssertionError: If `factor` is not an integer greater than or
        equal to 1, or if the weight tensor `w` does not have 4 dimensions.
    """

    assert isinstance(factor, int) and factor >= 1

    # Check weight shape.
    assert len(w.shape) == 4
    _, inC, convH, convW = w.shape

    assert convW == convH

    # Setup filter kernel.
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor**2))
    p = (k.shape[0] - factor) - (convW - 1)

    stride = (factor, factor)

    # Determine data dimensions.
    stride = [1, 1, factor, factor]
    output_shape = (
        (_shape(x, 2) - 1) * factor + convH,
        (_shape(x, 3) - 1) * factor + convW,
    )
    output_padding = (
        output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
        output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW,
    )
    assert output_padding[0] >= 0 and output_padding[1] >= 0
    num_groups = _shape(x, 1) // inC

    # Transpose weights.
    w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
    w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
    w = torch.reshape(w, (num_groups * inC, -1, convH, convW))

    x = F.conv_transpose2d(
        x, w, stride=stride, output_padding=output_padding, padding=0
    )

    return upfirdn2d(
        x, torch.tensor(k, device=x.device), pad=((p + 1) // 2 + factor - 1, p // 2 + 1)
    )


def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
    """
    Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

    This function performs a combined operation of a 2D convolution followed
    by downsampling. It applies the convolution with the provided weights and
    then downsamples the result using the specified FIR filter. The padding
    is performed only once at the beginning, making this approach more
    efficient than performing the operations separately.

    Args:
        x: Input tensor of shape `[N, C, H, W]` or `[N, H, W, C]`, where:
            - N is the batch size,
            - C is the number of channels,
            - H is the height,
            - W is the width of the input tensor.
        w: Weight tensor of shape `[filterH, filterW, inChannels, outChannels]`.
            Grouped convolution can be performed by setting `inChannels =
            x.shape[0] // numGroups`.
        k: FIR filter of shape `[firH, firW]` or `[firN]` (separable). The
            default is `[1] * factor`, which corresponds to average pooling.
        factor: Integer downsampling factor (default: 2).
        gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of shape `[N, C, H // factor, W // factor]` or
        `[N, H // factor, W // factor, C]`, with the same datatype as `x`.

    Examples:
        >>> x = torch.randn(1, 3, 64, 64)  # Example input
        >>> w = torch.randn(3, 3, 3, 3)    # Example weights
        >>> output = conv_downsample_2d(x, w, factor=2)
        >>> print(output.shape)
        torch.Size([1, 3, 32, 32])  # Output shape after downsampling
    """

    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW = w.shape
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    s = [factor, factor]
    x = upfirdn2d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2, p // 2))
    return F.conv2d(x, w, stride=s, padding=0)


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def _shape(x, dim):
    return x.shape[dim]


def upsample_2d(x, k=None, factor=2, gain=1):
    """
    Upsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and upsamples each image with the given filter. The filter is normalized so
    that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is
    padded with zeros so that its shape is a multiple of the upsampling factor.

    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
                      (separable). The default is `[1] * factor`, which
                      corresponds to nearest-neighbor upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]` or
        `[N, H * factor, W * factor, C]`, and same datatype as `x`.

    Examples:
        >>> x = torch.rand(1, 3, 64, 64)  # A random image batch
        >>> upsampled = upsample_2d(x, factor=2)
        >>> print(upsampled.shape)  # Should be [1, 3, 128, 128]

    Note:
        The function is designed to efficiently upsample images while maintaining
        the characteristics of the input data.

    Raises:
        AssertionError: If `factor` is not a positive integer.
    """
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor**2))
    p = k.shape[0] - factor
    return upfirdn2d(
        x,
        torch.tensor(k, device=x.device),
        up=factor,
        pad=((p + 1) // 2 + factor - 1, p // 2),
    )


def downsample_2d(x, k=None, factor=2, gain=1):
    """
        Downsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and downsamples each image with the given filter. The filter is normalized
    so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded
    with zeros so that its shape is a multiple of the downsampling factor.

    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
                      (separable). The default is `[1] * factor`, which corresponds
                      to average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]`.

    Examples:
        >>> import torch
        >>> x = torch.randn(1, 3, 64, 64)  # Example input tensor
        >>> downsampled = downsample_2d(x, factor=2)
        >>> print(downsampled.shape)  # Should output: torch.Size([1, 3, 32, 32])
    """

    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn2d(
        x, torch.tensor(k, device=x.device), down=factor, pad=((p + 1) // 2, p // 2)
    )
