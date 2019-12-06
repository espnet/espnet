import pytest
import torch

from espnet2.layers.utterance_mvn import UtteranceMVN


def test_repr():
    print(UtteranceMVN())


@pytest.mark.parametrize(
    "norm_vars, norm_means",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_forward(norm_vars, norm_means):
    layer = UtteranceMVN(norm_means=norm_means, norm_vars=norm_vars)
    x = torch.randn(2, 10, 80)
    y, _ = layer(x)
    assert y.shape == (2, 10, 80)
    y, ylen = layer(x, torch.tensor([10, 8], dtype=torch.long))
    assert (ylen == torch.tensor((10, 8), dtype=torch.long)).all()


@pytest.mark.parametrize(
    "norm_vars, norm_means",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_backward_leaf_in(norm_vars, norm_means):
    layer = UtteranceMVN(norm_means=norm_means, norm_vars=norm_vars)
    x = torch.randn(2, 1000, requires_grad=True)
    y, _ = layer(x)
    y.sum().backward()


@pytest.mark.parametrize(
    "norm_vars, norm_means",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_backward_not_leaf_in(norm_vars, norm_means):
    layer = UtteranceMVN(norm_means=norm_means, norm_vars=norm_vars)
    x = torch.randn(2, 1000, requires_grad=True)
    x = x + 2
    y, _ = layer(x)
    y.sum().backward()
