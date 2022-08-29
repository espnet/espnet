import pytest
import torch

from espnet2.enh.decoder.conv_decoder import ConvDecoder


@pytest.mark.parametrize("channel", [64])
@pytest.mark.parametrize("kernel_size", [10, 20])
@pytest.mark.parametrize("stride", [5, 10])
def test_ConvEncoder_backward(channel, kernel_size, stride):
    decoder = ConvDecoder(channel=channel, kernel_size=kernel_size, stride=stride,)

    x = torch.rand(2, 200, channel)
    x_lens = torch.tensor(
        [199 * stride + kernel_size, 199 * stride + kernel_size], dtype=torch.long
    )
    y, flens = decoder(x, x_lens)
    y.sum().backward()
