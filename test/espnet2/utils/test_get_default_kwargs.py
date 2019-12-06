from typing import Any

import pytest

from espnet2.utils.get_default_kwargs import get_default_kwargs


class Dummy:
    pass


def func1(a, b=3):
    pass


def func2(b=[{1, 2, 3}]):
    pass


def func3(b=dict(c=4), d=6.7):
    pass


def func4(b=Dummy()):
    pass


def func5(b={3: 5}):
    pass


def func6(b=(3, 5)):
    pass


def func7(b=(4, Dummy())):
    pass


@pytest.mark.parametrize(
    "func, desired",
    [
        (func1, {"b": 3}),
        (func2, {"b": [[1, 2, 3]]}),
        (func3, {"b": {"c": 4}, "d": 6.7}),
        (func4, {}),
        (func5, {}),
        (func6, {"b": [3, 5]}),
        (func7, {}),
    ],
)
def test_get_defaut_kwargs(func, desired: Any):
    assert get_default_kwargs(func) == desired
