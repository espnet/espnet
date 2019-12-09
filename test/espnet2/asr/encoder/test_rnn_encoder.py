import pytest
import torch

from espnet2.asr.encoder.rnn_encoder import Encoder


@pytest.mark.parametrize(
    "etype",
    ["lstm", "blstm", "bgru", "lstmp", "blstmp", "bgrup",
     "vgglstm", "vggblstm", "vggbgru",
     "vgglstmp", "vggblstmp", "vggbgrup",
     ])
@pytest.mark.parametrize("subsample", [None, (2, 2, 1, 1)])
def test_Encoder_forward_backward(etype, subsample):
    encoder = Encoder(5, etype=etype, subsample=subsample)
    x = torch.randn(2, 10, 5, requires_grad=True)
    x_lens = torch.LongTensor([10, 8])
    y, _, _ = encoder(x, x_lens)
    y.sum().backward()


def test_Encoder_out_dim():
    encoder = Encoder(5, eprojs=10)
    assert encoder.out_dim() == 10


def test_Encoder_invalid_type():
    with pytest.raises(ValueError):
        Encoder(5, etype='fff')
