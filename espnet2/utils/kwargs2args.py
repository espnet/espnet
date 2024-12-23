import inspect


def func(a: int, b, *, c, **kwargs):
    """
        Converts a dictionary of keyword arguments into a tuple of positional arguments
    for a given function.

    This utility function inspects the signature of the provided function and
    maps the keyword arguments to their respective positional parameters,
    returning them as a tuple. It supports both positional and keyword-only
    parameters.

    Args:
        func (Callable): The function whose signature will be inspected.
        kwargs (dict): A dictionary of keyword arguments to be converted.

    Returns:
        tuple: A tuple containing the positional arguments in the order defined
        by the function's signature. If some positional arguments are not provided
        in `kwargs`, they will be returned as `None`.

    Examples:
        >>> def example_func(x, y, *, z):
        ...     return x + y + z
        ...
        >>> kwargs = {'x': 1, 'y': 2, 'z': 3}
        >>> kwargs2args(example_func, kwargs)
        (1, 2)

        >>> kwargs = {'x': 10}
        >>> kwargs2args(example_func, kwargs)
        (10, None)

    Note:
        This function only maps the keyword arguments that match the function's
        parameters. If a keyword is not present in the function's signature, it
        will be ignored.
    """
    pass


def kwargs2args(func, kwargs):
    """
        Convert a dictionary of keyword arguments to a tuple of positional arguments
    for a given function.

    This function inspects the parameters of the provided function and maps the
    keyword arguments from the dictionary to the corresponding positional
    arguments. The resulting tuple will contain the positional arguments in the
    order defined by the function signature.

    Args:
        func (Callable): The function whose parameters will be inspected.
        kwargs (dict): A dictionary of keyword arguments to be converted.

    Returns:
        tuple: A tuple containing the positional arguments corresponding to the
        provided keyword arguments. The length of the tuple will be equal to the
        number of positional parameters in the function signature, but will only
        include arguments that were found in the `kwargs` dictionary.

    Examples:
        >>> def example_func(x, y, *, z):
        ...     pass
        >>> kwargs = {'x': 1, 'y': 2, 'z': 3}
        >>> kwargs2args(example_func, kwargs)
        (1, 2)

        >>> kwargs = {'x': 5}
        >>> kwargs2args(example_func, kwargs)
        (5,)

        >>> kwargs = {'z': 10}
        >>> kwargs2args(example_func, kwargs)
        (None, None)
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
