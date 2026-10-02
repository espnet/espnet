import pytest
import torch

from espnet2.enh.layers.wpe import wpe_one_iteration


def test_wpe_one_iteration_accepts_legacy_complextensor():
    torch_complex = pytest.importorskip("torch_complex")
    torch.random.manual_seed(0)
    Y = torch.complex(torch.randn(33, 2, 40), torch.randn(33, 2, 40))  # (F, C, T)
    Yl = torch_complex.tensor.ComplexTensor(Y.real, Y.imag)
    power = (Y.real**2 + Y.imag**2).mean(-2)  # (F, T)
    out_native = wpe_one_iteration(Y, power, taps=2, delay=1)
    out_legacy = wpe_one_iteration(Yl, power, taps=2, delay=1)
    # the legacy form is converted at the boundary and a native tensor comes back
    assert torch.is_complex(out_legacy)
    assert torch.allclose(out_native, out_legacy)
