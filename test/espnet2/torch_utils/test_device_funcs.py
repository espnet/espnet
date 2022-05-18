import dataclasses
from typing import NamedTuple

import pytest
import torch

from espnet2.torch_utils.device_funcs import force_gatherable, to_device

x = torch.tensor(10)


@dataclasses.dataclass(frozen=True)
class Data:
    x: torch.Tensor


class Named(NamedTuple):
    x: torch.Tensor


@pytest.mark.parametrize(
    "obj",
    [x, x.numpy(), (x,), [x], {"x": [x]}, {x}, Data(x), Named(x), 23, 3.0, None],
)
def test_to_device(obj):
    to_device(obj, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Require cuda")
def test_to_device_cuda():
    obj = {"a": [torch.tensor([0, 1])]}
    obj2 = to_device(obj, "cuda")
    assert obj2["a"][0].device == torch.device("cuda:0")


@pytest.mark.parametrize(
    "obj",
    [x, x.numpy(), (x,), [x], {"x": x}, {x}, Data(x), Named(x), 23, 3.0, None],
)
def test_force_gatherable(obj):
    force_gatherable(obj, "cpu")


def test_force_gatherable_0dim_to_1dim():
    obj = {"a": [3]}
    obj2 = force_gatherable(obj, "cpu")
    assert obj2["a"][0].shape == (1,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Require cuda")
def test_force_gatherable_cuda():
    obj = {"a": [torch.tensor([0, 1])]}
    obj2 = force_gatherable(obj, "cuda")
    assert obj2["a"][0].device == torch.device("cuda:0")
