import pytest
import torch

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding


@pytest.mark.parametrize(
    "dtype, device", [(dt, dv) for dt in ("float32", "float64") for dv in ("cpu", "cuda")])
def test_pe_extendable(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device is available")
    dtype = getattr(torch, dtype)
    dim = 2
    pe = PositionalEncoding(dim, 0.0, 3).to(dtype=dtype, device=device)
    x = torch.rand(2, 3, dim, dtype=dtype, device=device)
    y = pe(x)
    init_cache = pe.pe

    # test not extended from init
    x = torch.rand(2, 3, dim, dtype=dtype, device=device)
    y = pe(x)
    assert pe.pe is init_cache

    x = torch.rand(2, 5, dim, dtype=dtype, device=device)
    y = pe(x)

    sd = pe.state_dict()
    assert len(sd) == 0, "PositionalEncoding should save nothing"
    pe2 = PositionalEncoding(dim, 0.0, 3).to(dtype=dtype, device=device)
    pe2.load_state_dict(sd)
    y2 = pe2(x)
    assert torch.allclose(y, y2)


@pytest.mark.parametrize(
    "dtype, device", [(dt, dv) for dt in ("float32", "float64") for dv in ("cpu", "cuda")])
def test_scaled_pe_extendable(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device is available")
    dtype = getattr(torch, dtype)
    dim = 2
    pe = ScaledPositionalEncoding(dim, 0.0, 3).to(dtype=dtype, device=device)
    x = torch.rand(2, 3, dim, dtype=dtype, device=device)
    y = pe(x)
    init_cache = pe.pe

    # test not extended from init
    x = torch.rand(2, 3, dim, dtype=dtype, device=device)
    y = pe(x)
    assert pe.pe is init_cache

    x = torch.rand(2, 5, dim, dtype=dtype, device=device)
    y = pe(x)

    sd = pe.state_dict()
    assert sd == {"alpha": pe.alpha}, "ScaledPositionalEncoding should save only alpha"
    pe2 = ScaledPositionalEncoding(dim, 0.0, 3).to(dtype=dtype, device=device)
    pe2.load_state_dict(sd)
    y2 = pe2(x)
    assert torch.allclose(y, y2)
