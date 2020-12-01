import pytest
import torch

from espnet2.asr.encoder.rnn_encoder_mix import RNNEncoderMix


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("use_projection", [True, False])
@pytest.mark.parametrize("num_spkrs", [2, 3])
@pytest.mark.parametrize("subsample", [None, (2, 2, 1, 1)])
def test_Encoder_forward_backward(
    rnn_type, bidirectional, use_projection, num_spkrs, subsample
):
    encoder = RNNEncoderMix(
        5,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        use_projection=use_projection,
        num_layers_sd=2,
        num_layers_rec=2,
        num_spkrs=num_spkrs,
        hidden_size=50,
        output_size=50,
        subsample=subsample,
    )
    x = torch.randn(2, 10, 5, requires_grad=True)
    x_lens = torch.LongTensor([10, 8])
    y, _, _ = encoder(x, x_lens)
    assert len(y) == num_spkrs, len(y)
    sum(y).sum().backward()


def test_Encoder_output_size():
    encoder = RNNEncoderMix(5, output_size=10)
    assert encoder.output_size() == 10


def test_Encoder_invalid_type():
    with pytest.raises(ValueError):
        RNNEncoderMix(5, rnn_type="fff")
