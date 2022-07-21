import pytest
import torch

from espnet2.enh.encoder.conv_encoder import ConvEncoder


@pytest.mark.parametrize("channel", [64])
@pytest.mark.parametrize("kernel_size", [10, 20])
@pytest.mark.parametrize("stride", [5, 10])
def test_ConvEncoder_backward(channel, kernel_size, stride):
    encoder = ConvEncoder(channel=channel, kernel_size=kernel_size, stride=stride,)

    x = torch.rand(2, 32000)
    x_lens = torch.tensor([32000, 30000], dtype=torch.long)
    y, flens = encoder(x, x_lens)
    y.sum().backward()
