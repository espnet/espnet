# This code is derived from https://github.com/HazyResearch/state-spaces

"""Utilities for dealing with collection objects (lists, dicts) and configs."""
import functools
from typing import Callable, Mapping, Sequence

import hydra
from omegaconf import DictConfig, ListConfig


def is_list(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


def is_dict(x):
    return isinstance(x, Mapping)


def to_dict(x, recursive=True):
    """Convert Sequence or Mapping object to dict.

    lists get converted to {0: x[0], 1: x[1], ...}
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
    """Convert an object to list.

    If Sequence (e.g. list, tuple, Listconfig): just return it

    Special case: If non-recursive and not a list, wrap in list
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
    if obj is None:
        assert len(attrs) == 0
        return []
    return [getattr(obj, attr, None) for attr in attrs]


def instantiate(registry, config, *args, partial=False, wrap=None, **kwargs):
    """Instantiate registered module.

    registry: Dictionary mapping names to functions or target paths
            (e.g. {'model': 'models.SequenceModel'})
    config: Dictionary with a '_name_' key indicating which element of the registry
            to grab, and kwargs to be passed into the target constructor
    wrap: wrap the target class (e.g. ema optimizer or tasks.wrap)
    *args, **kwargs: additional arguments
            to override the config to pass into the target constructor
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
    breakpoint()
    return hydra.utils.get_class(path=registry[_name_])


def omegaconf_filter_keys(d, fn=None):
    """Only keep keys where fn(key) is True. Support nested DictConfig."""
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
