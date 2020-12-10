import torch

from espnet2.asr.frontend.sinc import LightweightSincConvs
from espnet2.asr.frontend.sinc import SpatialDropout


def test_spatial_dropout():
    dropout = SpatialDropout()
    x = torch.randn([5, 20, 1, 40], requires_grad=True)
    y = dropout(x)


def test_lightweight_sinc_convolutions_output_size():
    frontend = LightweightSincConvs()
    assert frontend.output_size() == frontend.get_odim()


def test_lightweight_sinc_convolutions_forward():
    frontend = LightweightSincConvs(fs="16000")
    x = torch.randn([2, 50, 1, 400], requires_grad=True)
    x_lengths = torch.LongTensor([30, 9])
    y, y_lengths = frontend(x, x_lengths)
    y.sum().backward()
