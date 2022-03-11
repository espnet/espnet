from pathlib import Path

import numpy as np
import pytest
import torch

from espnet2.layers.global_mvn import GlobalMVN


@pytest.fixture()
def stats_file(tmp_path: Path):
    """Kaldi like style"""
    p = tmp_path / "stats.npy"

    count = 10
    np.random.seed(0)
    x = np.random.randn(count, 80)
    s = x.sum(0)
    s = np.pad(s, [0, 1], mode="constant", constant_values=count)
    s2 = (x**2).sum(0)
    s2 = np.pad(s2, [0, 1], mode="constant", constant_values=0.0)

    stats = np.stack([s, s2])
    np.save(p, stats)
    return p


@pytest.fixture()
def stats_file2(tmp_path: Path):
    """New style"""
    p = tmp_path / "stats.npz"

    count = 10
    np.random.seed(0)
    x = np.random.randn(count, 80)
    s = x.sum(0)
    s2 = (x**2).sum(0)

    np.savez(p, sum=s, sum_square=s2, count=count)
    return p


@pytest.mark.parametrize(
    "norm_vars, norm_means",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_repl(stats_file, norm_vars, norm_means):
    layer = GlobalMVN(stats_file, norm_means=norm_means, norm_vars=norm_vars)
    print(layer)


@pytest.mark.parametrize(
    "norm_vars, norm_means",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_backward_leaf_in(stats_file, norm_vars, norm_means):
    layer = GlobalMVN(stats_file, norm_means=norm_means, norm_vars=norm_vars)
    x = torch.randn(1, 2, 80, requires_grad=True)
    y, _ = layer(x)
    y.sum().backward()


@pytest.mark.parametrize(
    "norm_vars, norm_means",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_backward_not_leaf_in(stats_file, norm_vars, norm_means):
    layer = GlobalMVN(stats_file, norm_means=norm_means, norm_vars=norm_vars)
    x = torch.randn(2, 3, 80, requires_grad=True)
    x = x + 2
    y, _ = layer(x)
    y.sum().backward()


@pytest.mark.parametrize(
    "norm_vars, norm_means",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_inverse_backwar_leaf_in(stats_file, norm_vars, norm_means):
    layer = GlobalMVN(stats_file, norm_means=norm_means, norm_vars=norm_vars)
    x = torch.randn(2, 3, 80, requires_grad=True)
    y, _ = layer.inverse(x)
    y.sum().backward()


@pytest.mark.parametrize(
    "norm_vars, norm_means",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_inverse_backwar_not_leaf_in(stats_file, norm_vars, norm_means):
    layer = GlobalMVN(stats_file, norm_means=norm_means, norm_vars=norm_vars)
    x = torch.randn(2, 3, 80, requires_grad=True)
    x = x + 2
    y, _ = layer.inverse(x)


@pytest.mark.parametrize(
    "norm_vars, norm_means",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_inverse_identity(stats_file, norm_vars, norm_means):
    layer = GlobalMVN(stats_file, norm_means=norm_means, norm_vars=norm_vars)
    x = torch.randn(2, 3, 80)
    y, _ = layer(x)
    x2, _ = layer.inverse(y)
    np.testing.assert_allclose(x.numpy(), x2.numpy())


@pytest.mark.parametrize(
    "norm_vars, norm_means",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_new_style_stats_file(stats_file, stats_file2, norm_vars, norm_means):
    layer = GlobalMVN(stats_file, norm_means=norm_means, norm_vars=norm_vars)
    layer2 = GlobalMVN(stats_file2, norm_means=norm_means, norm_vars=norm_vars)
    x = torch.randn(2, 3, 80)
    y, _ = layer(x)
    y2, _ = layer2(x)
    np.testing.assert_allclose(y.numpy(), y2.numpy())
