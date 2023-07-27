import pytest
import torch

from espnet2.asr.postencoder.length_adaptor_postencoder import LengthAdaptorPostEncoder


@pytest.mark.parametrize("input_layer, odim", [[None, None], ["linear", 100]])
def test_length_adaptor_forward(input_layer, odim):
    idim = 200
    x_max_length = 50
    length_adaptor_n_layers = 1
    postencoder = LengthAdaptorPostEncoder(
        idim, length_adaptor_n_layers, input_layer, odim
    )
    x = torch.randn([4, x_max_length, idim], requires_grad=True)
    x_lengths = torch.LongTensor([20, 5, x_max_length, 15])
    y, y_lengths = postencoder(x, x_lengths)
    y.sum().backward()

    if odim is None:
        odim_expected = idim
    else:
        odim_expected = odim

    assert postencoder.output_size() == odim_expected

    y_shape_1_expected = x_max_length // 2**length_adaptor_n_layers
    y_lengths_expected = (
        x_lengths.float().div(2**length_adaptor_n_layers).floor().long()
    )

    assert y.shape == torch.Size([4, y_shape_1_expected, odim_expected])
    assert torch.equal(y_lengths, y_lengths_expected)


def test_transformers_too_short_utt():
    idim = 400
    postencoder = LengthAdaptorPostEncoder(idim, 2)
    x = torch.randn([2, 3, idim], requires_grad=True)
    x_lengths = torch.LongTensor([3, 2])
    with pytest.raises(Exception):
        y, y_lengths = postencoder(x, x_lengths)
