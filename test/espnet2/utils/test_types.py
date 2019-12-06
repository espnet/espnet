from contextlib import contextmanager
from typing import Any

import pytest

from espnet2.utils.types import float_or_none
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2pair_str
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


@contextmanager
def pytest_raise_or_nothing(exception_or_any: Any):
    if isinstance(exception_or_any, type) and issubclass(
        exception_or_any, Exception
    ):
        with pytest.raises(exception_or_any):
            yield
    else:
        yield


@pytest.mark.parametrize(
    "value, desired",
    [
        ("true", True),
        ("false", False),
        ("True", True),
        ("False", False),
        ("aa", ValueError),
    ],
)
def test_str2bool(value: str, desired: Any):
    with pytest_raise_or_nothing(desired):
        assert str2bool(value) == desired


@pytest.mark.parametrize(
    "value, desired",
    [("3", 3), ("3 ", 3), ("none", None), ("aa", ValueError), ],
)
def test_int_or_none(value: str, desired: Any):
    with pytest_raise_or_nothing(desired):
        assert int_or_none(value) == desired


@pytest.mark.parametrize(
    "value, desired",
    [("3.5", 3.5), ("3.5 ", 3.5), ("none", None), ("aa", ValueError), ],
)
def test_float_or_none(value: str, desired: Any):
    with pytest_raise_or_nothing(desired):
        assert float_or_none(value) == desired


@pytest.mark.parametrize("value, desired", [("none", None), ("aa", "aa"), ])
def test_str_or_none(value: str, desired: Any):
    with pytest_raise_or_nothing(desired):
        assert str_or_none(value) == desired


@pytest.mark.parametrize(
    "value, desired",
    [("a, b", ("a", "b")), ("a,b,c", ValueError), ("a", ValueError), ],
)
def test_str2pair_str(value: str, desired: Any):
    with pytest_raise_or_nothing(desired):
        assert str2pair_str(value) == desired


@pytest.mark.parametrize(
    "value, desired",
    [("a,b, c", ("a", "b", "c")), ("a,b", ValueError), ("a", ValueError), ],
)
def test_str2triple_str(value: str, desired: Any):
    with pytest_raise_or_nothing(desired):
        assert str2triple_str(value) == desired
