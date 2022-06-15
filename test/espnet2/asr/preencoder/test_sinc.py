import torch

from espnet2.asr.preencoder.sinc import LightweightSincConvs, SpatialDropout


def test_spatial_dropout():
    dropout = SpatialDropout()
    x = torch.randn([5, 20, 40], requires_grad=True)
    y = dropout(x)
    assert x.shape == y.shape


def test_lightweight_sinc_convolutions_output_size():
    frontend = LightweightSincConvs()
    idim = 400
    # Get output dimension by making one inference.
    # The test vector that is used has dimensions (1, T, 1, idim).
    # T was set to idim without any special reason,
    in_test = torch.zeros((1, idim, 1, idim))
    out, _ = frontend.forward(in_test, [idim])
    odim = out.size(2)
    assert frontend.output_size() == odim


def test_lightweight_sinc_convolutions_forward():
    frontend = LightweightSincConvs(fs="16000")
    x = torch.randn([2, 50, 1, 400], requires_grad=True)
    x_lengths = torch.LongTensor([30, 9])
    y, y_lengths = frontend(x, x_lengths)
    y.sum().backward()
    assert y.shape == torch.Size([2, 50, 256])
