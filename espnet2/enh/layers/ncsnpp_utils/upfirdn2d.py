"""UpFirDn2d functions for upsampling, padding, FIR filter and downsampling.

Functions are ported from https://github.com/NVlabs/stylegan2.
"""

import torch
from torch.nn import functional as F


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """
    UpFirDn2d functions for upsampling, padding, FIR filter, and downsampling.

These functions are designed to perform a series of operations that include 
upsampling, applying a FIR filter, and downsampling in a two-dimensional 
space. The implementation is adapted from the StyleGAN2 repository 
(https://github.com/NVlabs/stylegan2).

Args:
    input (torch.Tensor): The input tensor of shape 
        (N, C, H, W) where N is the batch size, C is the number of 
        channels, H is the height, and W is the width.
    kernel (torch.Tensor): The FIR filter kernel tensor of shape 
        (KH, KW) where KH is the height and KW is the width of the 
        kernel.
    up (int, optional): The upsampling factor for both dimensions. 
        Default is 1.
    down (int, optional): The downsampling factor for both dimensions. 
        Default is 1.
    pad (tuple, optional): A tuple of two integers specifying the 
        padding (pad_y0, pad_y1) for height and (pad_x0, pad_x1) 
        for width. Default is (0, 0).

Returns:
    torch.Tensor: The output tensor after applying the upsampling, FIR 
    filter, and downsampling operations. The output shape is 
    (N, C, out_h, out_w) where out_h and out_w are calculated based 
    on the input dimensions and the specified factors.

Examples:
    >>> input_tensor = torch.randn(1, 3, 64, 64)  # Example input
    >>> kernel = torch.ones((3, 3))  # Example kernel
    >>> output_tensor = upfirdn2d(input_tensor, kernel, up=2, down=1, pad=(1, 1))
    >>> print(output_tensor.shape)  # Output shape will vary based on input

Note:
    This function assumes that the input tensor has at least four 
    dimensions (N, C, H, W). The kernel must be a 2D tensor.

Raises:
    ValueError: If the input tensor or kernel does not have the 
    expected dimensions.
    """
    out = upfirdn2d_native(
        input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
    )
    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    """
    UpFirDn2d functions for upsampling, padding, FIR filter, and downsampling.

This module provides the `upfirdn2d` and `upfirdn2d_native` functions, which are 
used for upsampling, padding, and downsampling operations on 2D input data 
with an optional FIR filter applied. The functions are ported from 
https://github.com/NVlabs/stylegan2.

The `upfirdn2d` function serves as a high-level interface that utilizes 
`upfirdn2d_native` for the actual processing.

Args:
    input (torch.Tensor): The input tensor of shape 
        (N, C, H, W) where N is the batch size, C is the number of channels,
        H is the height, and W is the width.
    kernel (torch.Tensor): The FIR filter kernel of shape (kH, kW).
    up (int, optional): Upsampling factor for both dimensions. Defaults to 1.
    down (int, optional): Downsampling factor for both dimensions. Defaults to 1.
    pad (tuple, optional): A tuple of two integers (pad_x, pad_y) for padding 
        in the width and height dimensions. Defaults to (0, 0).

Returns:
    torch.Tensor: The output tensor after applying upsampling, padding, 
    FIR filtering, and downsampling. The shape of the output is 
    (N, C, out_h, out_w) where out_h and out_w are computed based on the 
    input size and the operations performed.

Examples:
    >>> import torch
    >>> input_tensor = torch.randn(1, 3, 64, 64)  # Example input
    >>> kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])  # Example kernel
    >>> output = upfirdn2d(input_tensor, kernel, up=2, down=1, pad=(1, 1))
    >>> print(output.shape)  # Expected output shape: (1, 3, 128, 128)

Note:
    The kernel should be a 2D tensor and it is expected to have odd dimensions 
    for proper centering during convolution.

Raises:
    ValueError: If the input tensor has less than 3 dimensions or if the 
    kernel has an invalid shape.

Todo:
    - Add support for non-square kernels in future versions.
    """
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)
