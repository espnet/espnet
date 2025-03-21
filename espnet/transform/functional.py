"""Functional module."""

import inspect

from espnet.transform.transform_interface import TransformInterface
from espnet.utils.check_kwargs import check_kwargs


class FuncTrans(TransformInterface):
    """Functional Transformation.

    WARNING:
        Builtin or C/C++ functions may not work properly
        because this class heavily depends on the `inspect` module.

    Usage:

    >>> def foo_bar(x, a=1, b=2):
    ...     '''Foo bar
    ...     :param x: input
    ...     :param int a: default 1
    ...     :param int b: default 2
    ...     '''
    ...     return x + a - b


    >>> class FooBar(FuncTrans):
    ...     _func = foo_bar
    ...     __doc__ = foo_bar.__doc__
    """

    _func = None

    def __init__(self, **kwargs):
        """Initialize Class."""
        self.kwargs = kwargs
        check_kwargs(self.func, kwargs)

    def __call__(self, x):
        """Process call method."""
        return self.func(x, **self.kwargs)

    @classmethod
    def add_arguments(cls, parser):
        """Add arguments to parser."""
        fname = cls._func.__name__.replace("_", "-")
        group = parser.add_argument_group(fname + " transformation setting")
        for k, v in cls.default_params().items():
            # TODO(karita): get help and choices from docstring?
            attr = k.replace("_", "-")
            group.add_argument(f"--{fname}-{attr}", default=v, type=type(v))
        return parser

    @property
    def func(self):
        """Return type of self."""
        return type(self)._func

    @classmethod
    def default_params(cls):
        """Return default parameters."""
        try:
            d = dict(inspect.signature(cls._func).parameters)
        except ValueError:
            d = dict()
        return {
            k: v.default for k, v in d.items() if v.default != inspect.Parameter.empty
        }

    def __repr__(self):
        """Return string with details of class."""
        params = self.default_params()
        params.update(**self.kwargs)
        ret = self.__class__.__name__ + "("
        if len(params) == 0:
            return ret + ")"
        for k, v in params.items():
            ret += "{}={}, ".format(k, v)
        return ret[:-2] + ")"
