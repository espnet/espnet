# This code is derived from https://github.com/HazyResearch/state-spaces

"""Utilities for dealing with collection objects (lists, dicts) and configs."""
import functools
from typing import Callable, Mapping, Sequence

import hydra
from omegaconf import DictConfig, ListConfig


def is_list(x):
    """
    Determine if the input is a list-like object.

    This function checks whether the provided input `x` is an instance of 
    a sequence (like a list or tuple) but not a string. It is useful for 
    differentiating between sequence types in various data processing tasks.

    Args:
        x: The input object to check.

    Returns:
        bool: True if `x` is a sequence and not a string; False otherwise.

    Examples:
        >>> is_list([1, 2, 3])
        True
        >>> is_list((1, 2, 3))
        True
        >>> is_list("hello")
        False
        >>> is_list(123)
        False
    """
    return isinstance(x, Sequence) and not isinstance(x, str)


def is_dict(x):
    """
    Determines whether the given object is a dictionary-like structure.

This function checks if the provided input is an instance of a dictionary or 
any mapping type. It can be useful for type-checking when working with 
configuration data or other collections.

Args:
    x: The object to check.

Returns:
    bool: True if the object is a dictionary-like structure, False otherwise.

Examples:
    >>> is_dict({'key': 'value'})
    True
    >>> is_dict(['item1', 'item2'])
    False
    >>> is_dict(None)
    False
    >>> is_dict({'key1': 1, 'key2': 2})
    True
    """
    return isinstance(x, Mapping)


def to_dict(x, recursive=True):
    """
    Convert a Sequence or Mapping object to a dictionary.

    This function takes an input object `x`, which can be a list, tuple, 
    dictionary, or any other type. If `x` is a list or tuple, it is 
    converted to a dictionary with indices as keys. If `x` is a dictionary, 
    its values are recursively converted to dictionaries if `recursive` is 
    set to True. If `x` is neither a list nor a dictionary, it is returned 
    unchanged.

    Args:
        x (Sequence or Mapping): The input object to be converted.
        recursive (bool): If True, recursively convert nested 
            dictionaries/lists. Default is True.

    Returns:
        dict: A dictionary representation of the input object, 
            or the input itself if it is neither a list nor a dictionary.

    Examples:
        >>> to_dict([1, 2, 3])
        {0: 1, 1: 2, 2: 3}

        >>> to_dict({'a': 1, 'b': [2, 3]})
        {'a': 1, 'b': {0: 2, 1: 3}}

        >>> to_dict('string')
        'string'

        >>> to_dict({'a': 1, 'b': {'c': 2}}, recursive=True)
        {'a': 1, 'b': {'c': 2}}

    Note:
        The function distinguishes between lists, dictionaries, and other 
        types to ensure appropriate conversion.

    Raises:
        TypeError: If the input type is unsupported.
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

    This function converts the input object to a list format. If the input 
    is already a sequence (e.g., list, tuple, or ListConfig), it is returned 
    as is. If the input is a non-sequence object and `recursive` is set to 
    False, the object is wrapped in a list. If `recursive` is True, the 
    function will convert each element of the input sequence to a list.

    Args:
        x (Any): The object to convert to a list.
        recursive (bool): If True, apply conversion recursively to elements 
                          of the input sequence. Defaults to False.

    Returns:
        list: The converted list object.

    Examples:
        >>> to_list([1, 2, 3])
        [1, 2, 3]

        >>> to_list((1, 2, 3))
        [1, 2, 3]

        >>> to_list(5)
        [5]

        >>> to_list([1, (2, 3)], recursive=True)
        [1, [2, 3]]

        >>> to_list("not a sequence")
        ["not a sequence"]
    
    Note:
        This function treats strings as non-sequence objects. If the input 
        is a string, it will be wrapped in a list regardless of the 
        `recursive` parameter.
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

    This function retrieves the values of specified attributes from the given
    object. If the object is `None`, an empty list is returned. If an attribute
    does not exist on the object, `None` is returned for that attribute.

    Args:
        obj: The object from which to extract attributes. If `None`, return an
            empty list.
        *attrs: A variable number of attribute names to extract from the object.

    Returns:
        A list of values corresponding to the specified attributes. The list
        will contain `None` for any attributes that do not exist on the object.

    Examples:
        class Sample:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = "value2"

        sample_obj = Sample()

        # Extract existing attributes
        values = extract_attrs_from_obj(sample_obj, 'attr1', 'attr2', 'attr3')
        print(values)  # Output: ['value1', 'value2', None]

        # Extract attributes from a None object
        values_none = extract_attrs_from_obj(None, 'attr1', 'attr2')
        print(values_none)  # Output: []

    Note:
        This function does not raise an exception for missing attributes; it
        simply returns `None` for those attributes.
    """
    if obj is None:
        assert len(attrs) == 0
        return []
    return [getattr(obj, attr, None) for attr in attrs]


def instantiate(registry, config, *args, partial=False, wrap=None, **kwargs):
    """
    Instantiate a registered module from the given configuration.

    This function retrieves a callable from the provided registry using the 
    configuration dictionary. It can either directly instantiate the object 
    or return a partial function that can be called later. The configuration 
    dictionary should contain a key '_name_' which indicates the target 
    callable to instantiate.

    Args:
        registry (dict): A dictionary mapping names to functions or target 
            paths (e.g. {'model': 'models.SequenceModel'}).
        config (dict or str): A configuration dictionary that must contain 
            a '_name_' key to specify which element of the registry to use, 
            along with any additional keyword arguments for the target constructor.
        *args: Additional positional arguments to override the config and 
            be passed to the target constructor.
        partial (bool, optional): If True, returns a partial object instead 
            of instantiating it. Defaults to False.
        wrap (Callable, optional): A function to wrap the target class, 
            e.g., an EMA optimizer or a task wrapper. Defaults to None.
        **kwargs: Additional keyword arguments to override the config 
            and be passed to the target constructor.

    Returns:
        object: The instantiated object if `partial` is False, 
            otherwise a partial object.

    Raises:
        NotImplementedError: If the target retrieved from the registry is 
            neither a string nor a callable.

    Examples:
        # Example 1: Instantiating a model
        model_config = {
            '_name_': 'model_name',
            'param1': value1,
            'param2': value2
        }
        model = instantiate(registry, model_config)

        # Example 2: Creating a partial function
        optimizer_config = {
            '_name_': 'optimizer_name',
            'lr': 0.01
        }
        partial_optimizer = instantiate(registry, optimizer_config, partial=True)

    Note:
        Ensure that the '_name_' key in the config is correctly set to match 
        the keys in the registry.

    Todo:
        - Add more error handling for edge cases.
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
    Retrieve a class from the provided registry using its name.

    This function takes a registry dictionary that maps names to class paths and
    returns the class corresponding to the specified name. It utilizes Hydra's
    utility functions to resolve the class path.

    Args:
        registry (Mapping[str, str]): A mapping of names to class paths.
        _name_ (str): The name of the class to retrieve from the registry.

    Returns:
        type: The class corresponding to the provided name from the registry.

    Raises:
        KeyError: If the provided name does not exist in the registry.
        NotImplementedError: If the target path is not a string or callable.

    Examples:
        >>> registry = {
        ...     'MyClass': 'path.to.MyClass',
        ... }
        >>> MyClass = get_class(registry, 'MyClass')
        >>> instance = MyClass()
        
    Note:
        Ensure that the registry is populated with valid class paths before calling
        this function.
    """
    return hydra.utils.get_class(path=registry[_name_])


def omegaconf_filter_keys(d, fn=None):
    """
    Filter keys from a nested OmegaConf dictionary based on a given function.

    This function recursively traverses a nested OmegaConf structure (i.e., 
    `DictConfig` or `ListConfig`) and retains only those keys for which the 
    provided function `fn` returns `True`. If no function is provided, all keys 
    are retained.

    Args:
        d (Union[DictConfig, ListConfig]): The input OmegaConf structure to filter.
        fn (Callable[[str], bool], optional): A function that takes a key (string) 
            and returns a boolean. Defaults to a function that always returns True.

    Returns:
        Union[DictConfig, ListConfig]: A new OmegaConf structure containing only 
        the keys for which `fn(key)` is True.

    Examples:
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.create({'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}})
        >>> filtered = omegaconf_filter_keys(config, lambda k: k in ['a', 'c'])
        >>> print(filtered)
        {'a': 1, 'c': {'d': 3, 'e': 4}}

        >>> filtered = omegaconf_filter_keys(config)
        >>> print(filtered)
        {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}}

    Note:
        This function is particularly useful when working with complex 
        configuration files, allowing selective access to relevant parameters 
        based on specific criteria.
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
