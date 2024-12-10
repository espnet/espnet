import inspect


class Invalid:
    """
    Marker object for not serializable-object.

    This class serves as a marker to indicate that an object is not
    serializable. It can be used as a return value in functions that need
    to signal the presence of an invalid object, particularly in the context
    of serialization.

    Attributes:
        None

    Examples:
        >>> invalid_instance = Invalid()
        >>> isinstance(invalid_instance, Invalid)
        True
    """


def get_default_kwargs(func):
    """
        Get the default values of the input function.

    This function inspects the given function's parameters and retrieves the
    default values for each parameter. It also checks whether these values are
    serializable to YAML format. If a value is not serializable, it is replaced
    with a marker object.

    Args:
        func (callable): The function from which to retrieve default parameter
            values.

    Returns:
        dict: A dictionary containing the parameter names as keys and their
            corresponding default values. Non-serializable values are excluded
            from the returned dictionary.

    Raises:
        Invalid: A marker object indicating a non-serializable value.

    Examples:
        >>> def func(a, b=3):
        ...     pass
        >>> get_default_kwargs(func)
        {'b': 3}

    Note:
        This function uses the `inspect` module to analyze the function's
        signature and determine the default values.

    Todo:
        - Extend the functionality to handle more complex parameter types if
          necessary.
    """

    def yaml_serializable(value):
        # isinstance(x, tuple) includes namedtuple, so type is used here
        if type(value) is tuple:
            return yaml_serializable(list(value))
        elif isinstance(value, set):
            return yaml_serializable(list(value))
        elif isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                return Invalid
            retval = {}
            for k, v in value.items():
                v2 = yaml_serializable(v)
                # Register only valid object
                if v2 not in (Invalid, inspect.Parameter.empty):
                    retval[k] = v2
            return retval
        elif isinstance(value, list):
            retval = []
            for v in value:
                v2 = yaml_serializable(v)
                # If any elements in the list are invalid,
                # the list also becomes invalid
                if v2 is Invalid:
                    return Invalid
                else:
                    retval.append(v2)
            return retval
        elif value in (inspect.Parameter.empty, None):
            return value
        elif isinstance(value, (float, int, complex, bool, str, bytes)):
            return value
        else:
            return Invalid

    # params: An ordered mapping of inspect.Parameter
    params = inspect.signature(func).parameters
    data = {p.name: p.default for p in params.values()}
    # Remove not yaml-serializable object
    data = yaml_serializable(data)
    return data
