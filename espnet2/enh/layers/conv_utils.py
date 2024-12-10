# noqa: E501 ported from https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/7
import math


def num2tuple(num):
    """
    Convert a number into a tuple of two identical numbers.

    This utility function is useful for ensuring that a single integer
    input can be treated as a pair of values (e.g., for height and width
    in convolutional operations). If the input is already a tuple, it
    is returned unchanged.

    Args:
        num (int or tuple): A number or a tuple of two numbers.

    Returns:
        tuple: A tuple containing two identical numbers if `num` is an
        integer; otherwise, the original tuple if `num` is already a tuple.

    Examples:
        >>> num2tuple(5)
        (5, 5)

        >>> num2tuple((3, 4))
        (3, 4)

    Note:
        This function is primarily intended for internal use in
        convolution-related calculations.
    """
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Calculate the output shape of a 2D convolution layer.

    This function computes the height and width of the output tensor
    resulting from a 2D convolution operation based on the input
    dimensions, kernel size, stride, padding, and dilation. The
    function can handle both single integers and tuples for the
    parameters.

    Args:
        h_w (Union[int, tuple]): The height and width of the input tensor.
        kernel_size (Union[int, tuple], optional): The size of the kernel.
            Default is 1.
        stride (Union[int, tuple], optional): The stride of the convolution.
            Default is 1.
        pad (Union[int, tuple], optional): The padding applied to the input.
            Default is 0.
        dilation (Union[int, tuple], optional): The dilation factor for the
            kernel. Default is 1.

    Returns:
        tuple: A tuple containing the height and width of the output tensor.

    Examples:
        >>> conv2d_output_shape((32, 32), kernel_size=3, stride=1, pad=1)
        (32, 32)

        >>> conv2d_output_shape((32, 32), kernel_size=3, stride=2, pad=1)
        (16, 16)

    Note:
        The input dimensions (h_w) should be provided as a tuple (height,
        width) or as a single integer if both dimensions are the same.
    """
    h_w, kernel_size, stride, pad, dilation = (
        num2tuple(h_w),
        num2tuple(kernel_size),
        num2tuple(stride),
        num2tuple(pad),
        num2tuple(dilation),
    )
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = math.floor(
        (h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    w = math.floor(
        (h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    return h, w


def convtransp2d_output_shape(
    h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad=0
):
    """
        Calculate the output shape of a 2D transposed convolution layer.

    This function computes the height and width of the output tensor
    resulting from a 2D transposed convolution operation, given the input
    shape and various parameters such as kernel size, stride, padding,
    dilation, and output padding.

    Attributes:
        h_w (tuple): The height and width of the input tensor (H, W).
        kernel_size (tuple or int): The size of the convolution kernel (KH, KW).
        stride (tuple or int): The stride of the convolution (SH, SW).
        pad (tuple or int): The padding added to the input (PH, PW).
        dilation (tuple or int): The dilation rate for the kernel (DH, DW).
        out_pad (tuple or int): The additional output padding (OH, OW).

    Args:
        h_w (tuple or int): The input height and width.
        kernel_size (tuple or int, optional): The size of the convolution kernel.
            Default is 1.
        stride (tuple or int, optional): The stride of the convolution.
            Default is 1.
        pad (tuple or int, optional): The padding added to the input.
            Default is 0.
        dilation (tuple or int, optional): The dilation rate for the kernel.
            Default is 1.
        out_pad (tuple or int, optional): The additional output padding.
            Default is 0.

    Returns:
        tuple: The computed output height and width (H_out, W_out).

    Examples:
        >>> convtransp2d_output_shape((5, 5), kernel_size=3, stride=2, pad=1)
        (11, 11)

        >>> convtransp2d_output_shape((10, 10), kernel_size=(3, 3), stride=(2, 2),
        ...                            pad=(1, 1), dilation=(1, 1), out_pad=(1, 1))
        (22, 22)

    Note:
        The input height and width should be positive integers, and the
        kernel size, stride, padding, dilation, and output padding should
        be positive integers or tuples of two positive integers.

    Todo:
        - Add support for different data formats if required.
    """
    h_w, kernel_size, stride, pad, dilation, out_pad = (
        num2tuple(h_w),
        num2tuple(kernel_size),
        num2tuple(stride),
        num2tuple(pad),
        num2tuple(dilation),
        num2tuple(out_pad),
    )
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = (
        (h_w[0] - 1) * stride[0]
        - sum(pad[0])
        + dilation[0] * (kernel_size[0] - 1)
        + out_pad[0]
        + 1
    )
    w = (
        (h_w[1] - 1) * stride[1]
        - sum(pad[1])
        + dilation[1] * (kernel_size[1] - 1)
        + out_pad[1]
        + 1
    )

    return h, w
