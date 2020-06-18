import pytest
import torch

from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("use_projection", [True, False])
def test_Encoder_forward_backward(rnn_type, bidirectional, use_projection):
    encoder = VGGRNNEncoder(
        5, rnn_type=rnn_type, bidirectional=bidirectional, use_projection=use_projection
    )
    x = torch.randn(2, 10, 5, requires_grad=True)
    x_lens = torch.LongTensor([10, 8])
    y, _, _ = encoder(x, x_lens)
    y.sum().backward()


def test_Encoder_output_size():
    encoder = VGGRNNEncoder(5, output_size=10)
    assert encoder.output_size() == 10


def test_Encoder_invalid_type():
    with pytest.raises(ValueError):
        VGGRNNEncoder(5, rnn_type="fff")
