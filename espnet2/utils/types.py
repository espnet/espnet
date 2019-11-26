import argparse
from distutils.util import strtobool
from typing import Optional, Tuple

from typeguard import typechecked


@typechecked
def str2bool(value: str) -> bool:
    return bool(strtobool(value))


@typechecked
def int_or_none(value: Optional[str]) -> Optional[int]:
    """

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=int_or_none)
        >>> parser.parse_args(['--foo', '456'])
        Namespace(foo=456)
        >>> parser.parse_args(['--foo', 'none'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'null'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'nil'])
        Namespace(foo=None)

    """
    if value is None:
        return value
    if value.lower() in ('none', 'null', 'nil'):
        return None
    return int(value)


@typechecked
def float_or_none(value: Optional[str]) -> Optional[float]:
    """

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=float_or_none)
        >>> parser.parse_args(['--foo', '4.5'])
        Namespace(foo=4.5)
        >>> parser.parse_args(['--foo', 'none'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'null'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'nil'])
        Namespace(foo=None)

    """
    if value is None:
        return value
    if value.lower() in ('none', 'null', 'nil'):
        return None
    return float(value)


@typechecked
def str_or_none(value: Optional[str]) -> Optional[str]:
    """

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=str_or_none)
        >>> parser.parse_args(['--foo', 'aaa'])
        Namespace(foo='aaa')
        >>> parser.parse_args(['--foo', 'none'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'null'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'nil'])
        Namespace(foo=None)

    """
    if value is None:
        return value
    if value.lower() in ('none', 'null', 'nil'):
        return None
    return value


@typechecked
def str2pair_str(value: str) -> Tuple[str, str]:
    """

    Examples:
        >>> str2pair_str('abc,def ')
        ('abc', 'def')
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=str2pair_str)
        >>> parser.parse_args(['--foo', 'abc,def'])
        Namespace(foo=('abc', 'def'))

    """
    a, b = value.split(',')
    return a.strip(), b.strip()


@typechecked
def str2triple_str(value: str) -> Tuple[str, str, str]:
    """

    Examples:
        >>> str2triple_str('abc,def ,ghi')
        ('abc', 'def', 'ghi')
    """
    a, b, c = value.split(',')
    return a.strip(), b.strip(), c.strip()
