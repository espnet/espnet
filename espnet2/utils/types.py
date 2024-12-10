from distutils.util import strtobool
from typing import Optional, Tuple, Union

import humanfriendly


def str2bool(value: str) -> bool:
    """
        Converts a string representation of truth values to a boolean.

    This function utilizes the `strtobool` function from the `distutils.util`
    module to convert string values like 'y', 'n', 'true', 'false', '1', '0',
    etc., into their respective boolean representations.

    Args:
        value (str): The string value to be converted to boolean. It should be
            one of the common truthy or falsy string representations.

    Returns:
        bool: The boolean value corresponding to the input string.

    Examples:
        >>> str2bool('true')
        True
        >>> str2bool('false')
        False
        >>> str2bool('1')
        True
        >>> str2bool('0')
        False
        >>> str2bool('yes')
        True
        >>> str2bool('no')
        False

    Raises:
        ValueError: If the input string cannot be interpreted as a boolean.
    """
    return bool(strtobool(value))


def remove_parenthesis(value: str):
    """
        Removes parentheses or brackets from the beginning and end of a string.

    This function checks if the input string `value` starts and ends with
    parentheses `()` or brackets `[]`. If so, it removes them and returns
    the modified string. If the input string does not have these characters
    at both ends, it returns the string unchanged.

    Args:
        value (str): The input string from which to remove parentheses or
            brackets.

    Returns:
        str: The input string with outer parentheses or brackets removed.

    Examples:
        >>> remove_parenthesis("(example)")
        'example'
        >>> remove_parenthesis("[example]")
        'example'
        >>> remove_parenthesis("example")
        'example'
        >>> remove_parenthesis(" (example) ")
        'example'
    """
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]
    elif value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return value


def remove_quotes(value: str):
    """
        Remove surrounding quotes from a given string.

    This function takes a string input and removes leading and trailing quotes
    (if they exist). It supports both single and double quotes.

    Args:
        value (str): The input string from which to remove quotes.

    Returns:
        str: The input string without leading and trailing quotes.

    Examples:
        >>> remove_quotes('"Hello, World!"')
        'Hello, World!'

        >>> remove_quotes("'Python is great'")
        'Python is great'

        >>> remove_quotes("No quotes here")
        'No quotes here'

        >>> remove_quotes('"Quotes" and more "quotes"')
        'Quotes" and more "quotes'
    """
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    elif value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value


def int_or_none(value: str) -> Optional[int]:
    """
        Convert a string to an integer or return None if the string is a null value.

    This function checks if the input string represents a null value ("none",
    "null", or "nil"). If so, it returns None. Otherwise, it attempts to convert
    the string to an integer and returns the result.

    Args:
        value (str): The input string to be converted to an integer.

    Returns:
        Optional[int]: The converted integer if the string can be converted,
        or None if the string is a null value.

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
    Convert a string to a float or return None.

    This function attempts to convert a given string value to a float.
    If the string is one of "none", "null", or "nil" (case insensitive),
    it returns None. Otherwise, it converts the string to a float.

    Args:
        value (str): The string value to convert.

    Returns:
        Optional[float]: The converted float value, or None if the input
        string is "none", "null", or "nil".

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
    """
        Parse a human-friendly size string or return None if the input is not valid.

    This function takes a string input representing a size (e.g., "1 KB", "2.5 MB")
    and converts it into a float representing the size in bytes. If the input is
    one of the specified strings that represent null values ("none", "null", "nil"),
    the function will return None.

    Args:
        value (str): The size string to be parsed.

    Returns:
        Optional[float]: The size in bytes as a float, or None if the input is
        invalid.

    Examples:
        >>> humanfriendly_parse_size_or_none("1 KB")
        1024.0
        >>> humanfriendly_parse_size_or_none("2.5 MB")
        2621440.0
        >>> humanfriendly_parse_size_or_none("none")
        None
        >>> humanfriendly_parse_size_or_none("null")
        None
        >>> humanfriendly_parse_size_or_none("nil")
        None
    """
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return humanfriendly.parse_size(value)


def str_or_int(value: str) -> Union[str, int]:
    """
        Converts a string to an integer if possible; otherwise, returns the string.

    This function attempts to convert the input string `value` to an integer. If the
    conversion is successful, the integer is returned. If a ValueError is raised during
    the conversion, the original string is returned instead.

    Args:
        value (str): The string to convert to an integer.

    Returns:
        Union[str, int]: The converted integer if successful, otherwise the original
        string.

    Examples:
        >>> str_or_int("123")
        123
        >>> str_or_int("abc")
        'abc'
        >>> str_or_int("456")
        456
        >>> str_or_int("789.0")
        '789.0'
    """
    try:
        return int(value)
    except ValueError:
        return value


def str_or_none(value: str) -> Optional[str]:
    """
        Convert a string to None if it represents a null value.

    This function checks if the input string is one of the specified null
    representations ('none', 'null', 'nil') and returns None if it is.
    Otherwise, it returns the original string.

    Args:
        value (str): The input string to check and potentially convert.

    Returns:
        Optional[str]: Returns None if the input string represents a null
        value; otherwise, returns the original string.

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
        Convert a string representation of a pair into a tuple of strings.

    This function takes a string formatted as 'string1,string2' and converts it
    into a tuple containing the two strings. It also handles any surrounding
    parentheses or quotes that may be present in the input string.

    Args:
        value (str): A string representing a pair, formatted as
            'string1,string2'.

    Returns:
        Tuple[str, str]: A tuple containing the two strings extracted from the
            input.

    Examples:
        >>> import argparse
        >>> str2pair_str('abc,def ')
        ('abc', 'def')
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=str2pair_str)
        >>> parser.parse_args(['--foo', 'abc,def'])
        Namespace(foo=('abc', 'def'))

    Note:
        This function also trims any whitespace around the input strings and
        removes any enclosing parentheses or quotes.

    Raises:
        ValueError: If the input string does not contain exactly one comma,
            resulting in an inability to split the string into two parts.
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
    """
        Converts a comma-separated string into a tuple of three strings.

    This function takes a string formatted as "a,b,c" and returns a tuple
    containing the three components as strings. Leading and trailing spaces
    are removed, and any surrounding parentheses or quotes are stripped
    from each component.

    Args:
        value (str): A comma-separated string to be converted into a tuple.

    Returns:
        Tuple[str, str, str]: A tuple containing three strings parsed from the
        input value.

    Examples:
        >>> str2triple_str('abc,def ,ghi')
        ('abc', 'def', 'ghi')
        >>> str2triple_str('( abc, "def", ghi )')
        ('abc', 'def', 'ghi')

    Note:
        If the input string does not contain exactly two commas, this function
        will raise a ValueError.

    Todo:
        - Implement additional validation to ensure the input string contains
        exactly three elements.
    """
    value = remove_parenthesis(value)
    a, b, c = value.split(",")

    # Workaround for configargparse issues:
    # If the list values are given from yaml file,
    # the value givent to type() is shaped as python-list,
    # e.g. ['a', 'b', 'c'],
    # so we need to remove quotes from it.
    return remove_quotes(a), remove_quotes(b), remove_quotes(c)
