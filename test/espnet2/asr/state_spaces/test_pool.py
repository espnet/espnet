import pytest
import torch

from espnet2.asr.state_spaces.pool import (
    DownSample,
    UpSample,
    downsample,
    registry,
    upsample,
)


def _make_input(d_input, length, transposed):
    if transposed:
        return torch.randn(2, d_input, length)
    return torch.randn(2, length, d_input)


@pytest.mark.parametrize("transposed", [True, False])
@pytest.mark.parametrize("stride, expand", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_downsample_forward_matches_functional(transposed, stride, expand):
    d_input, length = 4, 8
    layer = DownSample(d_input, stride=stride, expand=expand, transposed=transposed)
    x = _make_input(d_input, length, transposed)
    assert torch.equal(layer(x), downsample(x, stride, expand, transposed))


@pytest.mark.parametrize("transposed", [True, False])
@pytest.mark.parametrize("stride, expand", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_upsample_forward_matches_functional(transposed, stride, expand):
    d_input, length = 4, 8
    layer = UpSample(d_input, stride=stride, expand=expand, transposed=transposed)
    x = _make_input(d_input, length, transposed)
    assert torch.equal(layer(x), upsample(x, stride, expand, transposed))


@pytest.mark.parametrize("transposed", [True, False])
def test_downsample_expand_honors_transposed(transposed):
    # `transposed` decides which axis `expand` repeats on, so it has to reach
    # `downsample` instead of being replaced by a literal.
    d_input, length, expand = 4, 8, 2
    layer = DownSample(d_input, expand=expand, transposed=transposed)
    y = layer(_make_input(d_input, length, transposed))
    if transposed:
        assert y.shape == (2, d_input * expand, length)
    else:
        assert y.shape == (2, length, d_input * expand)
    assert layer.d_output == d_input * expand


def test_registry_sample_entry_runs():
    layer = registry["sample"](4, stride=2)
    assert layer(torch.randn(2, 4, 8)).shape == (2, 4, 4)


def test_downsample_none_input():
    assert DownSample(4, stride=2)(None) is None
