import dataclasses
from argparse import Namespace

import pytest

from espnet2.utils.build_dataclass import build_dataclass


@dataclasses.dataclass
class A:
    a: str
    b: str


def test_build_dataclass():
    args = Namespace(a="foo", b="bar")
    a = build_dataclass(A, args)
    assert a.a == args.a
    assert a.b == args.b


def test_build_dataclass_insufficient():
    args = Namespace(a="foo")
    with pytest.raises(ValueError):
        build_dataclass(A, args)
