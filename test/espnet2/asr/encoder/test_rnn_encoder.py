import pytest
import torch

from espnet2.asr.encoder.rnn_encoder import RNNEncoder


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("use_projection", [True, False])
@pytest.mark.parametrize("subsample", [None, (2, 2, 1, 1)])
def test_Encoder_forward_backward(rnn_type, bidirectional, use_projection, subsample):
    encoder = RNNEncoder(
        5,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        use_projection=use_projection,
        subsample=subsample,
    )
    x = torch.randn(2, 10, 5, requires_grad=True)
    x_lens = torch.LongTensor([10, 8])
    y, _, _ = encoder(x, x_lens)
    y.sum().backward()


def test_Encoder_output_size():
    encoder = RNNEncoder(5, output_size=10)
    assert encoder.output_size() == 10


def test_Encoder_invalid_type():
    with pytest.raises(ValueError):
        RNNEncoder(5, rnn_type="fff")
