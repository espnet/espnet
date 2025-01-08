import pytest
import torch

from espnet2.speechlm.tokenizer.beats_tokenizer import (
    BeatsTokenizer,
    BeatsTokenizerConfig,
)


def test_tokenizer_encode():
    tokenizer_config = BeatsTokenizerConfig()
    tokenizer_config.encoder_layers = 2
    tokenizer_config.quant_n = 15
    tokenizer = BeatsTokenizer(tokenizer_config=vars(tokenizer_config))
    x = torch.randn(2, 16000)
    x_len = torch.LongTensor([16000, 12000])
    token_ids = tokenizer.encode(xs_pad=x, ilens=x_len)
    assert token_ids.shape[0] == 2
    assert token_ids.shape[1] == 48
    assert token_ids.min() >= 0
    assert token_ids.max() < 15
