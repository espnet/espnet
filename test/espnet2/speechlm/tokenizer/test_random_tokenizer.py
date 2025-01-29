import pytest
import torch

from espnet2.speechlm.tokenizer.random_tokenizer import RandomProjectionQuantizer


@pytest.mark.parametrize("n_codes", [5, 20, 1024])
def test_tokenizer_encode(n_codes):
    tokenizer = RandomProjectionQuantizer(
        dim=128, codebook_size=n_codes, codebook_dim=64
    )
    x = torch.randn(20, 10, 128)
    token_ids = tokenizer(x)
    assert token_ids.shape[:2] == x.shape[:2]
    assert token_ids.min() >= 0
    assert token_ids.max() < n_codes


def test_reproducibility():
    torch.manual_seed(42)
    model1 = RandomProjectionQuantizer(dim=128, codebook_size=512, codebook_dim=64)
    torch.manual_seed(42)
    model2 = RandomProjectionQuantizer(dim=128, codebook_size=512, codebook_dim=64)
    x = torch.randn(4, 10, 128)
    assert torch.equal(model1(x), model2(x)), "Outputs differ with the same seed!"
