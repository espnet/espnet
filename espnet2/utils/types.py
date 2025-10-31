"""Utils/Types."""

from distutils.util import strtobool
from typing import Optional, Tuple, Union

import humanfriendly


def str2bool(value: str) -> bool:
    """Convert string representation to boolean.

    Converts string values like 'true', 'false', '1', '0', 'yes', 'no', etc.
    to their corresponding boolean values.

    Args:
        value: String representation of a boolean value.

    Returns:
        Boolean value corresponding to the input string.

    Examples:
        >>> str2bool('true')
        True
        >>> str2bool('false')
        False
        >>> str2bool('1')
        True
        >>> str2bool('0')
        False

    """
    return bool(strtobool(value))


def remove_parenthesis(value: str):
    """Remove outer parenthesis or brackets from a string.

    Removes outer parenthesis '()' or brackets '[]' from the input string
    if they wrap the entire value. Leading and trailing whitespace is
    stripped before checking.

    Args:
        value: Input string that may have parenthesis or brackets.

    Returns:
        String with outer parenthesis or brackets removed if present,
        otherwise returns the string as-is (after stripping whitespace).

    Examples:
        >>> remove_parenthesis('(hello)')
        'hello'
        >>> remove_parenthesis('[world]')
        'world'
        >>> remove_parenthesis('  (test)  ')
        'test'
        >>> remove_parenthesis('no_brackets')
        'no_brackets'

    """
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]
    elif value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return value


def remove_quotes(value: str):
    """Remove outer quotes from a string.

    Removes outer double quotes '""' or single quotes '''  from the input
    string if they wrap the entire value. Leading and trailing whitespace
    is stripped before checking.

    Args:
        value: Input string that may have quotes.

    Returns:
        String with outer quotes removed if present, otherwise returns
        the string as-is (after stripping whitespace).

    Examples:
        >>> remove_quotes('"hello"')
        'hello'
        >>> remove_quotes("'world'")
        'world'
        >>> remove_quotes('  "test"  ')
        'test'
        >>> remove_quotes('no_quotes')
        'no_quotes'

    """
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    elif value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value


def int_or_none(value: str) -> Optional[int]:
    """int_or_none.

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
    """float_or_none.

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


def humanfriendly_parse_size_or_none(value) -> Optional[float]:
    """Parse a human-friendly file size string or return None if special value.

    Parses size strings like "1GB", "512MB", "1KB" using humanfriendly library.
    Returns None if the value is a special string ('none', 'null', or 'nil').

    Args:
        value: Human-friendly size string or special string representing None.

    Returns:
        Parsed size in bytes as a float, or None if value is 'none', 'null',
        or 'nil' (case-insensitive).

    Examples:
        >>> humanfriendly_parse_size_or_none('1GB')
        1000000000.0
        >>> humanfriendly_parse_size_or_none('512MB')
        512000000.0
        >>> humanfriendly_parse_size_or_none('none')
        None
        >>> humanfriendly_parse_size_or_none('null')
        None

    """
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return humanfriendly.parse_size(value)


def str_or_int(value: str) -> Union[str, int]:
    """Convert string to int if possible, otherwise return as string.

    Attempts to convert the input string to an integer. If the conversion
    fails (e.g., for non-numeric strings), returns the original string.
    Useful for flexible argument parsing where values can be either numeric
    or textual.

    Args:
        value: Input string that may represent an integer.

    Returns:
        Integer if the string can be converted to an integer, otherwise
        the original string value.

    Examples:
        >>> str_or_int('123')
        123
        >>> str_or_int('hello')
        'hello'
        >>> str_or_int('456xyz')
        '456xyz'

    """
    try:
        return int(value)
    except ValueError:
        return value


def str_or_none(value: str) -> Optional[str]:
    """str_or_none.

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
    """str2pair_str.

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

    # Workaround for configargparse issues:
    # If the list values are given from yaml file,
    # the value givent to type() is shaped as python-list,
    # e.g. ['a', 'b', 'c'],
    # so we need to remove double quotes from it.
    return remove_quotes(a), remove_quotes(b)


def str2triple_str(value: str) -> Tuple[str, str, str]:
    """str2triple_str.

    Examples:
        >>> str2triple_str('abc,def ,ghi')
        ('abc', 'def', 'ghi')
    """
    value = remove_parenthesis(value)
    a, b, c = value.split(",")

    # Workaround for configargparse issues:
    # If the list values are given from yaml file,
    # the value givent to type() is shaped as python-list,
    # e.g. ['a', 'b', 'c'],
    # so we need to remove quotes from it.
    return remove_quotes(a), remove_quotes(b), remove_quotes(c)
