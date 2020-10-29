import pytest
import torch

from espnet2.layers.time_warp import TimeWarp


@pytest.mark.parametrize("x_lens", [None, torch.tensor([80, 78])])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_TimeWarp(x_lens, requires_grad):
    time_warp = TimeWarp(window=10)
    x = torch.randn(2, 100, 80, requires_grad=requires_grad)
    y, y_lens = time_warp(x, x_lens)
    if x_lens is not None:
        assert all(l1 == l2 for l1, l2 in zip(x_lens, y_lens))
    if requires_grad:
        y.sum().backward()


def test_TimeWarp_repr():
    time_warp = TimeWarp(window=10)
    print(time_warp)
