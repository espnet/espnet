import pytest
import torch

from espnet2.enh.layers.conv_utils import conv2d_output_shape
from espnet2.enh.layers.conv_utils import convtransp2d_output_shape


@pytest.mark.parametrize("input_dim", [(10, 17), (10, 33)])
@pytest.mark.parametrize("kernel_size", [(1, 3), (3, 5)])
@pytest.mark.parametrize("stride", [(1, 1), (1, 2)])
@pytest.mark.parametrize("padding", [(0, 0), (0, 1)])
@pytest.mark.parametrize("dilation", [(1, 1), (1, 2)])
def test_conv2d_output_shape(input_dim, kernel_size, stride, padding, dilation):
    h, w = conv2d_output_shape(
        input_dim,
        kernel_size=kernel_size,
        stride=stride,
        pad=padding,
        dilation=dilation,
    )
    conv = torch.nn.Conv2d(
        1, 1, kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    x = torch.rand(1, 1, *input_dim)
    assert conv(x).shape[2:] == (h, w)


@pytest.mark.parametrize("input_dim", [(10, 17), (10, 33)])
@pytest.mark.parametrize("kernel_size", [(1, 3), (3, 5)])
@pytest.mark.parametrize("stride", [(1, 1), (1, 2)])
@pytest.mark.parametrize("padding", [(0, 0), (0, 1)])
@pytest.mark.parametrize("output_padding", [(0, 0), (0, 1)])
@pytest.mark.parametrize("dilation", [(1, 1), (1, 2)])
def test_deconv2d_output_shape(
    input_dim, kernel_size, stride, padding, output_padding, dilation
):
    if (
        output_padding[0] >= stride[0]
        or output_padding[0] >= dilation[0]
        or output_padding[1] >= stride[1]
        or output_padding[1] >= dilation[1]
    ):
        # skip invalid cases
        return
    h, w = convtransp2d_output_shape(
        input_dim,
        kernel_size=kernel_size,
        stride=stride,
        pad=padding,
        dilation=dilation,
        out_pad=output_padding,
    )
    deconv = torch.nn.ConvTranspose2d(
        1,
        1,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )
    x = torch.rand(1, 1, *input_dim)
    assert deconv(x).shape[2:] == (h, w)
