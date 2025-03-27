import pytest
import torch

from espnet2.cls.decoder.linear_decoder import LinearDecoder


@pytest.mark.execution_timeout(30)
@pytest.mark.parametrize("vocab_size", [10, 5])
@pytest.mark.parametrize("encoder_output_size", [4, 21])
@pytest.mark.parametrize("pooling", ["mean", "max", "CLS"])
@pytest.mark.parametrize("dropout", [0.1, 0.0])
def test_forward_backward(vocab_size, encoder_output_size, pooling, dropout):
    decoder = LinearDecoder(vocab_size, encoder_output_size, pooling, dropout)
    x = torch.randn(2, 10, encoder_output_size, requires_grad=True)
    x_len = torch.randint(1, 10, [2], dtype=torch.long)
    logits = decoder(x, x_len)
    assert logits.shape == (2, vocab_size), logits.shape
    logits.sum().backward()


@pytest.mark.execution_timeout(30)
@pytest.mark.parametrize("vocab_size", [10, 5])
@pytest.mark.parametrize("encoder_output_size", [4, 21])
@pytest.mark.parametrize("pooling", ["mean", "max", "CLS"])
@pytest.mark.parametrize("dropout", [0.1, 0.0])
def test_score(vocab_size, encoder_output_size, pooling, dropout):
    decoder = LinearDecoder(vocab_size, encoder_output_size, pooling, dropout)
    x = torch.randn(10, encoder_output_size)
    score, _ = decoder.score(ys=None, state=None, x=x)
    assert score.shape == (vocab_size,), score.shape


@pytest.mark.execution_timeout(30)
@pytest.mark.parametrize("vocab_size", [10, 5])
@pytest.mark.parametrize("encoder_output_size", [4, 21])
@pytest.mark.parametrize("pooling", ["mean", "max", "CLS"])
@pytest.mark.parametrize("dropout", [0.1, 0.0])
def test_output_size(vocab_size, encoder_output_size, pooling, dropout):
    decoder = LinearDecoder(vocab_size, encoder_output_size, pooling, dropout)
    assert (
        decoder.output_size() == vocab_size
    ), f"Decoder output size must match vocab size."
