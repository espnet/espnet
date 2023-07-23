import pytest
import torch

from espnet2.asr.postencoder.length_adaptor_postencoder import LengthAdaptorPostEncoder


@pytest.mark.parametrize("odim", [200, 400])
def test_length_adaptor_forward(odim):
    idim = 400
    length_adaptor_n_layers = 1
    postencoder = LengthAdaptorPostEncoder(idim, length_adaptor_n_layers, odim)
    x = torch.randn([4, 50, idim], requires_grad=True)
    x_lengths = torch.LongTensor([20, 5, 50, 15])
    y, y_lengths = postencoder(x, x_lengths)
    y.sum().backward()

    assert postencoder.output_size() == odim

    y_shape_1_expected = 50 // 2**length_adaptor_n_layers
    y_lengths_expected = (
        x_lengths.float().div(2**length_adaptor_n_layers).floor().long()
    )

    assert y.shape == torch.Size([4, y_shape_1_expected, odim])
    assert torch.equal(y_lengths, y_lengths_expected)


def test_transformers_too_short_utt():
    idim = 400
    postencoder = LengthAdaptorPostEncoder(idim, 2, idim)
    x = torch.randn([2, 3, idim], requires_grad=True)
    x_lengths = torch.LongTensor([3, 2])
    with pytest.raises(Exception):
        y, y_lengths = postencoder(x, x_lengths)
