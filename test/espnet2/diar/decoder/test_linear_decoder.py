import pytest
import torch

from espnet2.diar.decoder.linear_decoder import LinearDecoder


@pytest.mark.parametrize("encoder_output_size", [10])
@pytest.mark.parametrize("num_spk", [2])
def test_linear_decoder(encoder_output_size, num_spk):
    linear_decoder = LinearDecoder(
        encoder_output_size=encoder_output_size, num_spk=num_spk
    )
    input = torch.rand(5, 100, encoder_output_size)
    ilens = torch.tensor([100, 100, 100, 100, 100])
    output = linear_decoder.forward(input=input, ilens=ilens)
    assert output.shape == (5, 100, num_spk)
