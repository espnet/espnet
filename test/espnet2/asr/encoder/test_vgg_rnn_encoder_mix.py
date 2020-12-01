import pytest
import torch

from espnet2.asr.encoder.vgg_rnn_encoder_mix import VGGRNNEncoderMix


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("use_projection", [True, False])
@pytest.mark.parametrize("num_spkrs", [2, 3])
def test_Encoder_forward_backward(rnn_type, bidirectional, use_projection, num_spkrs):
    encoder = VGGRNNEncoderMix(
        5,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        use_projection=use_projection,
        num_spkrs=num_spkrs,
        hidden_size=20,
        output_size=20,
    )
    x = torch.randn(2, 10, 5, requires_grad=True)
    x_lens = torch.LongTensor([10, 8])
    y, _, _ = encoder(x, x_lens)
    assert len(y) == num_spkrs, len(y)
    sum(y).sum().backward()


def test_Encoder_output_size():
    encoder = VGGRNNEncoderMix(5, output_size=10)
    assert encoder.output_size() == 10


def test_Encoder_invalid_type():
    with pytest.raises(ValueError):
        VGGRNNEncoderMix(5, rnn_type="fff")
