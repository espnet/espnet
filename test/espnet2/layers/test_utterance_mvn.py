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


@pytest.mark.parametrize("norm_vars", [True, False])
def test_forward_excludes_padded_frames(norm_vars):
    layer = UtteranceMVN(norm_means=True, norm_vars=norm_vars)
    x = torch.randn(1, 6, 4) + 3.0
    padded = torch.cat([x, torch.zeros(1, 4, 4)], dim=1)
    ilens = torch.tensor([6], dtype=torch.long)

    y_padded, _ = layer(padded, ilens)
    y, _ = layer(x, ilens)

    # How much padding a batch happens to carry must not change the result.
    assert torch.allclose(y_padded[:, :6], y, atol=1e-6)
    assert (y_padded[:, 6:] == 0.0).all()


def test_forward_unit_variance_with_padding():
    layer = UtteranceMVN(norm_means=True, norm_vars=True)
    x = torch.cat([torch.randn(1, 6, 4) + 3.0, torch.zeros(1, 4, 4)], dim=1)

    y, _ = layer(x, torch.tensor([6], dtype=torch.long))

    std = y[0, :6].std(dim=0, unbiased=False)
    assert torch.allclose(std, torch.ones(4), atol=1e-5)


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
