from distutils.util import strtobool
from typing import Optional
from typing import Tuple


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def remove_parenthesis(value: str):
    value = value.strip()
    if value.startswith('(') or value.startswith('['):
        value = value[1:]
    if value.endswith(')') or value.endswith(']'):
        value = value[:-1]
    return value


def int_or_none(value: str) -> Optional[int]:
    """

    Examples:
        >>> import argparse
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
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return int(value)


def float_or_none(value: str) -> Optional[float]:
    """

    Examples:
        >>> import argparse
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
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return float(value)


def str_or_none(value: str) -> Optional[str]:
    """

    Examples:
        >>> import argparse
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
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return value


def str2pair_str(value: str) -> Tuple[str, str]:
    """

    Examples:
        >>> import argparse
        >>> str2pair_str('abc,def ')
        ('abc', 'def')
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=str2pair_str)
        >>> parser.parse_args(['--foo', 'abc,def'])
        Namespace(foo=('abc', 'def'))

    """
    value = remove_parenthesis(value)
    a, b = value.split(",")
    return a.strip(), b.strip()


def str2triple_str(value: str) -> Tuple[str, str, str]:
    """

    Examples:
        >>> str2triple_str('abc,def ,ghi')
        ('abc', 'def', 'ghi')
    """
    value = remove_parenthesis(value)
    a, b, c = value.split(",")
    return a.strip(), b.strip(), c.strip()
