import dataclasses
from typing import NamedTuple

import pytest
import torch

from espnet2.utils.device_funcs import to_device, force_gatherable

x = torch.tensor(10)


@dataclasses.dataclass(frozen=True)
class Data:
    x: torch.Tensor


class Named(NamedTuple):
    x: torch.Tensor


@pytest.mark.parametrize(
    'obj',
    [x,
     x.numpy(),
     (x,),
     [x],
     {'x': [x]},
     {x},
     Data(x),
     Named(x),
     23,
     3.,
     None,
     ])
def test_to_device(obj):
    to_device(obj, 'cpu')


@pytest.mark.parametrize(
    'obj',
    [x,
     x.numpy(),
     (x,),
     [x],
     {'x': x},
     {x},
     Data(x),
     Named(x),
     23,
     3.,
     None,
     ])
def test_force_gatherable(obj):
    force_gatherable(obj, 'cpu')
