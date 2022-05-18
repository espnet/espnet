import torch

from espnet2.asr.preencoder.linear import LinearProjection


def test_linear_projection_forward():
    idim = 400
    odim = 80
    preencoder = LinearProjection(input_size=idim, output_size=odim)
    x = torch.randn([2, 50, idim], requires_grad=True)
    x_lengths = torch.LongTensor([30, 15])
    y, y_lengths = preencoder(x, x_lengths)
    y.sum().backward()
    assert y.shape == torch.Size([2, 50, odim])
    assert torch.equal(y_lengths, x_lengths)
