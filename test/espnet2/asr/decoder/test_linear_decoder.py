import pytest
import torch

from espnet2.asr.decoder.linear_decoder import LinearDecoder


@pytest.mark.execution_timeout(30)
@pytest.mark.parametrize("vocab_size", [10, 5])
@pytest.mark.parametrize("encoder_output_size", [4, 21])
@pytest.mark.parametrize("pooling", ["mean", "max", "CLS"])
def test_forward_backward(vocab_size, encoder_output_size, pooling):
    decoder = LinearDecoder(vocab_size, encoder_output_size, pooling)
    x = torch.randn(2, 10, encoder_output_size, requires_grad=True)
    x_len = torch.randint(1, 10, [2], dtype=torch.long)
    logits = decoder(x, x_len)
    assert logits.shape == (2, vocab_size - 3), logits.shape
    logits.sum().backward()


@pytest.mark.execution_timeout(30)
@pytest.mark.parametrize("vocab_size", [10, 5])
@pytest.mark.parametrize("encoder_output_size", [4, 21])
@pytest.mark.parametrize("pooling", ["mean", "max", "CLS"])
def test_score(vocab_size, encoder_output_size, pooling):
    decoder = LinearDecoder(vocab_size, encoder_output_size, pooling)
    x = torch.randn(10, encoder_output_size)
    score, _ = decoder.score(ys=None, state=None, x=x)
    assert score.shape == (vocab_size,), score.shape
