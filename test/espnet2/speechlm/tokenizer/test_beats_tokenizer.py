import pytest
import torch

from espnet2.speechlm.tokenizer.beats_tokenizer import (
    BeatsTokenizer,
    BeatsTokenizerConfig,
)


@pytest.mark.parametrize("n_codes", [15, 20, 200])
def test_tokenizer_encode(n_codes):
    tokenizer_config = BeatsTokenizerConfig()
    tokenizer_config.encoder_layers = 2
    tokenizer_config.quant_n = n_codes
    tokenizer = BeatsTokenizer(tokenizer_config=vars(tokenizer_config))
    x = torch.randn(2, 16000)
    x_len = torch.LongTensor([16000, 12000])
    token_ids = tokenizer.encode(xs_pad=x, ilens=x_len)
    assert token_ids.shape[0] == 2
    assert token_ids.min() >= 0
    assert token_ids.max() < n_codes
