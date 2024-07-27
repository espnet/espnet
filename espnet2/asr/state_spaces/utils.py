# This code is derived from https://github.com/HazyResearch/state-spaces

"""Utilities for dealing with collection objects (lists, dicts) and configs."""
import functools
from typing import Callable, Mapping, Sequence

import hydra
from omegaconf import DictConfig, ListConfig


def is_list(x):
    """
    Check if the input is a list-like sequence (excluding strings).

    This function determines whether the input is a sequence (like a list or tuple)
    but not a string.

    Args:
        x: The object to be checked.

    Returns:
        bool: True if the input is a list-like sequence, False otherwise.

    Examples:
        >>> is_list([1, 2, 3])
        True
        >>> is_list((1, 2, 3))
        True
        >>> is_list("abc")
        False
        >>> is_list(123)
        False
    """
    return isinstance(x, Sequence) and not isinstance(x, str)


def is_dict(x):
    """
    Check if the input is a dictionary-like object.

    This function determines whether the input is an instance of a Mapping,
    which includes dictionaries and other mapping types.

    Args:
        x: The object to be checked.

    Returns:
        bool: True if the input is a dictionary-like object, False otherwise.

    Examples:
        >>> is_dict({'a': 1, 'b': 2})
        True
        >>> is_dict(dict(a=1, b=2))
        True
        >>> is_dict([1, 2, 3])
        False
        >>> is_dict("abc")
        False
    """
    return isinstance(x, Mapping)


def to_dict(x, recursive=True):
    """
    Convert a Sequence or Mapping object to a dictionary.

    This function converts list-like objects to dictionaries with integer keys,
    and recursively converts nested structures if specified.

    Args:
        x: The object to be converted to a dictionary.
        recursive (bool, optional): If True, recursively convert nested structures.
            Defaults to True.

    Returns:
        dict: The converted dictionary.

    Examples:
        >>> to_dict([1, 2, 3])
        {0: 1, 1: 2, 2: 3}
        >>> to_dict({'a': [1, 2], 'b': {'c': 3}}, recursive=True)
        {'a': {0: 1, 1: 2}, 'b': {'c': 3}}
        >>> to_dict({'a': [1, 2], 'b': {'c': 3}}, recursive=False)
        {'a': [1, 2], 'b': {'c': 3}}
        >>> to_dict(5)
        5

    Note:
        - Lists are converted to dictionaries with integer keys.
        - If the input is neither a Sequence nor a Mapping, it is returned unchanged.
    """
    if is_list(x):
        x = {i: v for i, v in enumerate(x)}
    if is_dict(x):
        if recursive:
            return {k: to_dict(v, recursive=recursive) for k, v in x.items()}
        else:
            return dict(x)
    else:
        return x


def to_list(x, recursive=False):
    """
    Convert an object to a list.

    This function converts Sequence objects (except strings) to lists and handles
    recursive conversion of nested structures if specified.

    Args:
        x: The object to be converted to a list.
        recursive (bool, optional): If True, recursively convert nested structures.
            Defaults to False.

    Returns:
        list: The converted list.

    Examples:
        >>> to_list((1, 2, 3))
        [1, 2, 3]
        >>> to_list([1, 2, 3])
        [1, 2, 3]
        >>> to_list("abc")
        ["abc"]
        >>> to_list(5)
        [5]
        >>> to_list([1, [2, 3], 4], recursive=True)
        [1, [2, 3], 4]
        >>> to_list([1, [2, 3], 4], recursive=False)
        [1, [2, 3], 4]

    Note:
        - If the input is not a Sequence and recursive is False, it will be wrapped in a list.
        - If recursive is True and the input is not a Sequence, it will be returned unchanged.
        - Strings are treated as non-Sequence objects and will be wrapped in a list if recursive is False.
    """
    if is_list(x):
        if recursive:
            return [to_list(_x) for _x in x]
        else:
            return list(x)
    else:
        if recursive:
            return x
        else:
            return [x]


def extract_attrs_from_obj(obj, *attrs):
    """
    Extract specified attributes from an object.

    This function retrieves the values of specified attributes from the given object.
    If the object is None, it returns an empty list.

    Args:
        obj: The object from which to extract attributes.
        *attrs: Variable length argument list of attribute names to extract.

    Returns:
        list: A list containing the values of the specified attributes.
              If an attribute doesn't exist, None is used as its value.

    Raises:
        AssertionError: If obj is None and attrs is not empty.

    Examples:
        >>> class Example:
        ...     def __init__(self):
        ...         self.a = 1
        ...         self.b = "test"
        >>> obj = Example()
        >>> extract_attrs_from_obj(obj, "a", "b", "c")
        [1, "test", None]
        >>> extract_attrs_from_obj(None)
        []

    Note:
        - If obj is None, the function expects no attributes to be specified.
        - For non-existent attributes, None is used as the value in the returned list.
    """
    if obj is None:
        assert len(attrs) == 0
        return []
    return [getattr(obj, attr, None) for attr in attrs]


def instantiate(registry, config, *args, partial=False, wrap=None, **kwargs):
    """
    Instantiate a registered module based on the provided configuration.

    This function creates an instance of a registered module using the provided
    configuration and additional arguments. It supports various instantiation
    scenarios and can optionally wrap the target class.

    Args:
        registry (dict): A dictionary mapping names to functions or target paths.
        config (dict or str): Configuration for instantiation. If a string, it's
            treated as the key for the registry. If a dict, it should contain a
            '_name_' key indicating which element of the registry to use.
        *args: Additional positional arguments to pass to the constructor.
        partial (bool, optional): If True, returns a partial function instead of
            an instantiated object. Defaults to False.
        wrap (callable, optional): A function to wrap the target class. Defaults to None.
        **kwargs: Additional keyword arguments to override the config and pass
            to the target constructor.

    Returns:
        object or functools.partial: The instantiated object or a partial function,
        depending on the 'partial' parameter.

    Raises:
        NotImplementedError: If the instantiate target is neither a string nor callable.

    Examples:
        >>> registry = {'model': 'models.SequenceModel'}
        >>> config = {'_name_': 'model', 'hidden_size': 128}
        >>> model = instantiate(registry, config)
        >>> partial_model = instantiate(registry, config, partial=True)
        >>> wrapped_model = instantiate(registry, config, wrap=some_wrapper_function)

    Note:
        - The function supports both string-based and callable-based registry entries.
        - If 'config' is a string, it's used as the key for the registry.
        - The '_name_' key is restored in the config after instantiation.
    """
    # Case 1: no config
    if config is None:
        return None
    # Case 2a: string means _name_ was overloaded
    if isinstance(config, str):
        _name_ = None
        _target_ = registry[config]
        config = {}
    # Case 2b: grab the desired callable from name
    else:
        _name_ = config.pop("_name_")
        _target_ = registry[_name_]

    # Retrieve the right constructor automatically based on type
    if isinstance(_target_, str):
        fn = hydra.utils.get_method(path=_target_)
    elif isinstance(_target_, Callable):
        fn = _target_
    else:
        raise NotImplementedError("instantiate target must be string or callable")

    # Instantiate object
    if wrap is not None:
        fn = wrap(fn)
    obj = functools.partial(fn, *args, **config, **kwargs)

    # Restore _name_
    if _name_ is not None:
        config["_name_"] = _name_

    if partial:
        return obj
    else:
        return obj()


def get_class(registry, _name_):
    """
    Retrieve a class from the registry based on the provided name.

    This function uses Hydra's get_class utility to fetch the class specified
    by the _name_ parameter from the given registry.

    Args:
        registry (dict): A dictionary mapping names to class paths.
        _name_ (str): The name of the class to retrieve from the registry.

    Returns:
        type: The class object corresponding to the specified name.

    Raises:
        Any exceptions raised by hydra.utils.get_class.

    Examples:
        >>> registry = {'MyClass': 'path.to.MyClass'}
        >>> MyClass = get_class(registry, 'MyClass')
        >>> instance = MyClass()

    Note:
        - This function includes a breakpoint() call, which will pause execution
          when the function is called. This is likely for debugging purposes and
          should be removed in production code.
        - The function relies on Hydra's get_class utility, which dynamically
          imports and returns the specified class.
    """
    return hydra.utils.get_class(path=registry[_name_])


def omegaconf_filter_keys(d, fn=None):
    """
    Filter keys in an OmegaConf structure based on a given function.

    This function traverses through a nested OmegaConf structure (DictConfig or ListConfig)
    and filters keys based on the provided function. It supports recursive filtering
    for nested structures.

    Args:
        d (Union[ListConfig, DictConfig, Any]): The OmegaConf structure to filter.
        fn (Callable[[str], bool], optional): A function that takes a key as input
            and returns True if the key should be kept, False otherwise.
            If None, all keys are kept. Defaults to None.

    Returns:
        Union[ListConfig, DictConfig, Any]: A new OmegaConf structure with filtered keys.

    Examples:
        >>> from omegaconf import DictConfig, ListConfig
        >>> config = DictConfig({'a': 1, 'b': ListConfig([1, 2]), 'c': DictConfig({'d': 3, 'e': 4})})
        >>> filtered = omegaconf_filter_keys(config, lambda k: k != 'b')
        >>> print(filtered)
        {'a': 1, 'c': {'d': 3, 'e': 4}}

        >>> def filter_func(key):
        ...     return not key.startswith('_')
        >>> config = DictConfig({'a': 1, '_b': 2, 'c': DictConfig({'d': 3, '_e': 4})})
        >>> filtered = omegaconf_filter_keys(config, filter_func)
        >>> print(filtered)
        {'a': 1, 'c': {'d': 3}}

    Note:
        - If no filter function is provided, all keys are kept.
        - The function preserves the OmegaConf structure (DictConfig for dictionaries,
          ListConfig for lists) in the returned object.
        - For non-dict and non-list inputs, the original input is returned unchanged.
    """
    if fn is None:

        def fn(_):
            return True

    if is_list(d):
        return ListConfig([omegaconf_filter_keys(v, fn) for v in d])
    elif is_dict(d):
        return DictConfig(
            {k: omegaconf_filter_keys(v, fn) for k, v in d.items() if fn(k)}
        )
    else:
        return d
