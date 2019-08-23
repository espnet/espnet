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


class LegacyPositionalEncoding(torch.nn.Module):
    """Positional encoding module until v.0.5.2."""

    def __init__(self, d_model, dropout_rate, max_len=5000):
        import math
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.max_len = max_len
        self.xscale = math.sqrt(d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LegacyScaledPositionalEncoding(PositionalEncoding):
    """Positional encoding module until v.0.5.2."""

    def __init__(self, d_model, dropout_rate, max_len=5000):
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, :x.size(1)]
        return self.dropout(x)


def test_compatibility():
    """Regression test for #1121"""
    x = torch.rand(2, 3, 4)

    legacy_net = torch.nn.Sequential(
        LegacyPositionalEncoding(4, 0.0),
        torch.nn.Linear(4, 2)
    )

    latest_net = torch.nn.Sequential(
        PositionalEncoding(4, 0.0),
        torch.nn.Linear(4, 2)
    )

    latest_net.load_state_dict(legacy_net.state_dict())
    legacy = legacy_net(x)
    latest = latest_net(x)
    assert torch.allclose(legacy, latest)

    legacy_net = torch.nn.Sequential(
        LegacyScaledPositionalEncoding(4, 0.0),
        torch.nn.Linear(4, 2)
    )

    latest_net = torch.nn.Sequential(
        ScaledPositionalEncoding(4, 0.0),
        torch.nn.Linear(4, 2)
    )

    latest_net.load_state_dict(legacy_net.state_dict())
    legacy = legacy_net(x)
    latest = latest_net(x)
    assert torch.allclose(legacy, latest)
