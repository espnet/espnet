from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.preencoder.sinc import SpatialDropout
import torch


def test_spatial_dropout():
    dropout = SpatialDropout()
    x = torch.randn([5, 20, 40], requires_grad=True)
    y = dropout(x)
    assert x.shape == y.shape


def test_lightweight_sinc_convolutions_output_size():
    frontend = LightweightSincConvs()
    assert frontend.output_size() == frontend.get_odim()


def test_lightweight_sinc_convolutions_forward():
    frontend = LightweightSincConvs(fs="16000")
    x = torch.randn([2, 50, 1, 400], requires_grad=True)
    x_lengths = torch.LongTensor([30, 9])
    y, y_lengths = frontend(x, x_lengths)
    y.sum().backward()
    assert y.shape == torch.Size([2, 50, 256])
