import pytest
import torch

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding


def test_pe_extendable():
    dim = 2
    pe = PositionalEncoding(dim, 0.0, 3)
    init_cache = pe.pe

    # test not extended from init
    x = torch.rand(2, 3, dim)
    y = pe(x)
    assert pe.pe is init_cache

    x = torch.rand(2, 5, dim)
    y = pe(x)
    assert x.shape == y.shape

    sd = pe.state_dict()
    assert len(sd) == 0, "PositionalEncoding should save nothing"
    pe2 = PositionalEncoding(dim, 0.0, 3)
    pe2.load_state_dict(sd)
    y2 = pe2(x)
    assert torch.all(y == y2)


def test_scaled_pe_extendable():
    dim = 2
    pe = ScaledPositionalEncoding(dim, 0.0, 3)
    init_cache = pe.pe

    # test not extended from init
    x = torch.rand(2, 3, dim)
    y = pe(x)
    assert pe.pe is init_cache

    x = torch.rand(2, 5, dim)
    y = pe(x)
    assert x.shape == y.shape

    sd = pe.state_dict()
    assert sd == {"alpha": pe.alpha}, "ScaledPositionalEncoding should save only alpha"
    pe2 = ScaledPositionalEncoding(dim, 0.0, 3)
    pe2.load_state_dict(sd)
    y2 = pe2(x)
    assert torch.all(y == y2)
