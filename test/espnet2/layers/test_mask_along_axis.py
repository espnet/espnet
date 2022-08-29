import pytest
import torch

from espnet2.layers.mask_along_axis import MaskAlongAxis


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("replace_with_zero", [False, True])
@pytest.mark.parametrize("dim", ["freq", "time"])
def test_MaskAlongAxis(dim, replace_with_zero, requires_grad):
    freq_mask = MaskAlongAxis(
        dim=dim, mask_width_range=30, num_mask=2, replace_with_zero=replace_with_zero,
    )
    x = torch.randn(2, 100, 80, requires_grad=requires_grad)
    x_lens = torch.tensor([80, 78])
    y, y_lens = freq_mask(x, x_lens)
    assert all(l1 == l2 for l1, l2 in zip(x_lens, y_lens))
    if requires_grad:
        y.sum().backward()


@pytest.mark.parametrize("replace_with_zero", [False, True])
@pytest.mark.parametrize("dim", ["freq", "time"])
def test_MaskAlongAxis_repr(dim, replace_with_zero):
    freq_mask = MaskAlongAxis(
        dim=dim, mask_width_range=30, num_mask=2, replace_with_zero=replace_with_zero,
    )
    print(freq_mask)
