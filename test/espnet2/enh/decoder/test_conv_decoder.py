import pytest
import torch

from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder


@pytest.mark.parametrize("channel", [64])
@pytest.mark.parametrize("kernel_size", [10, 20])
@pytest.mark.parametrize("stride", [5, 10])
def test_ConvEncoder_backward(channel, kernel_size, stride):
    decoder = ConvDecoder(
        channel=channel,
        kernel_size=kernel_size,
        stride=stride,
    )

    x = torch.rand(2, 200, channel)
    x_lens = torch.tensor(
        [199 * stride + kernel_size, 199 * stride + kernel_size], dtype=torch.long
    )
    y, flens = decoder(x, x_lens)
    y.sum().backward()


@pytest.mark.parametrize("channel", [64])
@pytest.mark.parametrize("kernel_size", [10, 20])
@pytest.mark.parametrize("stride", [5, 10])
def test_conv_dec_streaming(channel, kernel_size, stride):
    input_audio = torch.randn((1, 100))
    ilens = torch.LongTensor([100])

    encoder = ConvEncoder(kernel_size=kernel_size, stride=stride, channel=channel)
    decoder = ConvDecoder(kernel_size=kernel_size, stride=stride, channel=channel)
    frames, flens = encoder(input_audio, ilens)
    wav, ilens = decoder(frames, ilens)

    splited = encoder.streaming_frame(input_audio)

    sframes = [encoder.forward_streaming(s) for s in splited]
    swavs = [decoder.forward_streaming(s) for s in sframes]
    merged = decoder.streaming_merge(swavs, ilens)

    sframes = torch.cat(sframes, dim=1)

    torch.testing.assert_allclose(sframes, frames)
    torch.testing.assert_allclose(wav, merged)
