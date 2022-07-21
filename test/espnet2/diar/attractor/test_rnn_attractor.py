import pytest
import torch

from espnet2.diar.attractor.rnn_attractor import RnnAttractor


@pytest.mark.parametrize("encoder_output_size", [10])
@pytest.mark.parametrize("layer", [1])
@pytest.mark.parametrize("unit", [10])
@pytest.mark.parametrize("dropout", [0.1])
def test_rnn_attractor(encoder_output_size, layer, unit, dropout):
    eda = RnnAttractor(
        encoder_output_size=encoder_output_size,
        layer=layer,
        unit=unit,
        dropout=dropout,
    )
    enc_input = torch.rand(5, 100, encoder_output_size)
    ilens = torch.tensor([100, 100, 100, 100, 100])
    dec_input = torch.zeros(5, 3, encoder_output_size)
    attractor, att_prob = eda.forward(
        enc_input=enc_input, ilens=ilens, dec_input=dec_input,
    )
    assert attractor.shape == (5, 3, encoder_output_size)
    assert att_prob.shape == (5, 3, 1)
