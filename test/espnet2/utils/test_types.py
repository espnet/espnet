from contextlib import contextmanager
from typing import Any

import pytest

from espnet2.utils.types import (
    float_or_none,
    humanfriendly_parse_size_or_none,
    int_or_none,
    remove_parenthesis,
    str2bool,
    str2pair_str,
    str2triple_str,
    str_or_int,
    str_or_none,
)


@contextmanager
def pytest_raise_or_nothing(exception_or_any: Any):
    if isinstance(exception_or_any, type) and issubclass(exception_or_any, Exception):
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
    "value, desired", [("3", 3), ("3 ", 3), ("none", None), ("aa", ValueError)],
)
def test_int_or_none(value: str, desired: Any):
    with pytest_raise_or_nothing(desired):
        assert int_or_none(value) == desired


@pytest.mark.parametrize(
    "value, desired", [("3.5", 3.5), ("3.5 ", 3.5), ("none", None), ("aa", ValueError)],
)
def test_float_or_none(value: str, desired: Any):
    with pytest_raise_or_nothing(desired):
        assert float_or_none(value) == desired


@pytest.mark.parametrize(
    "value, desired", [("3k", 3000), ("2m ", 2000000), ("none", None)],
)
def test_humanfriendly_parse_size_or_none(value: str, desired: Any):
    with pytest_raise_or_nothing(desired):
        assert humanfriendly_parse_size_or_none(value) == desired


@pytest.mark.parametrize(
    "value, desired", [("3", 3), ("3 ", 3), ("aa", "aa")],
)
def test_str_or_int(value: str, desired: Any):
    assert str_or_int(value) == desired


@pytest.mark.parametrize("value, desired", [("none", None), ("aa", "aa")])
def test_str_or_none(value: str, desired: Any):
    with pytest_raise_or_nothing(desired):
        assert str_or_none(value) == desired


@pytest.mark.parametrize(
    "value, desired",
    [
        ("a, b", ("a", "b")),
        ("a,b,c", ValueError),
        ("a", ValueError),
        ("['a', 'b']", ("a", "b")),
    ],
)
def test_str2pair_str(value: str, desired: Any):
    with pytest_raise_or_nothing(desired):
        assert str2pair_str(value) == desired


@pytest.mark.parametrize(
    "value, desired",
    [
        ("a,b, c", ("a", "b", "c")),
        ("a,b", ValueError),
        ("a", ValueError),
        ("['a', 'b', 'c']", ("a", "b", "c")),
    ],
)
def test_str2triple_str(value: str, desired: Any):
    with pytest_raise_or_nothing(desired):
        assert str2triple_str(value) == desired


@pytest.mark.parametrize(
    "value, desired", [(" (a v c) ", "a v c"), ("[ 0999 ]", " 0999 ")]
)
def test_remove_parenthesis(value: str, desired: Any):
    assert remove_parenthesis(value) == desired
