"""Convert keyword arguments to positional arguments.

This module provides utilities for converting keyword arguments (kwargs) to positional
arguments based on a function's signature. It's useful for adapting function calls when
you have arguments in dictionary form but need them in positional form.
"""

import inspect


def func(a: int, b, *, c, **kwargs):
    """Test example for kwargs2args.

    This is a sample function used to demonstrate the kwargs2args conversion.
    It has positional parameters (a, b), a keyword-only parameter (c),
    and accepts additional keyword arguments (**kwargs).

    Args:
        a: A required integer parameter.
        b: A required positional parameter.
        c: A keyword-only parameter.
        **kwargs: Additional keyword arguments.

    """
    pass


def kwargs2args(func, kwargs):
    """Convert keyword arguments to positional arguments based on function signature.

    This function inspects the signature of a given function and converts keyword
    arguments into positional arguments. It processes kwargs in the order of the
    function's parameters and returns a tuple of positional arguments up to the
    first None value (i.e., up to the first parameter not provided in kwargs).

    Args:
        func: The function whose signature will be inspected.
        kwargs: A dictionary of keyword arguments to convert.

    Returns:
        A tuple of positional arguments in the order of the function's parameters.
        The tuple includes all consecutive arguments from the start, stopping at
        the first parameter that was not provided in kwargs.

    Examples:
        >>> def example_func(a, b, c):
        ...     pass
        >>> kwargs2args(example_func, {'a': 1, 'b': 2, 'c': 3})
        (1, 2, 3)
        >>> kwargs2args(example_func, {'a': 1, 'b': 2})
        (1, 2)
        >>> kwargs2args(example_func, {'a': 1, 'c': 3})
        (1,)

    """
    parameters = inspect.signature(func).parameters
    d = {k: i for i, k in enumerate(parameters)}
    args = [None for i in range(len(parameters))]
    for k, v in kwargs.items():
        if k in d:
            args[d[k]] = v

    for i, v in enumerate(args):
        if v is None:
            break

    return tuple(args[:i])
