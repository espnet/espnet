import argparse
import copy
from distutils.util import strtobool
from typing import Optional, Tuple

import yaml
from typeguard import typechecked


@typechecked
def str2bool(value: str) -> bool:
    return bool(strtobool(value))


@typechecked
def int_or_none(value: Optional[str]) -> Optional[int]:
    if value is None:
        return value
    if value.lower() in ('none', 'null', 'nil'):
        return None
    return int(value)


@typechecked
def str_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return value
    if value.lower() in ('none', 'null', 'nil'):
        return None
    return value


@typechecked
def str2tuple_str(value: str) -> Tuple[str, str]:
    """

    Examples:
        >>> str2tuple_str('abc,def ')
        ('abc', 'def')
    """
    a, b = value.split(',')
    return a.strip(), b.strip()


@typechecked
def str2triple_str(value: str) -> Tuple[str, str, str]:
    """

    Examples:
        >>> str2triple_str('abc,def ,ghi')
        ('abc', 'def', 'ghi')
    """
    a, b, c = value.split(',')
    return a.strip(), b.strip(), c.strip()


class NestedDictAction(argparse.Action):
    """

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--conf', action=NestedDictAction,
        ...                         default={'a': 4})
        >>> parser.parse_args(['--conf', 'a=3', '--conf', 'c=4'])
        Namespace(conf={'a': 3, 'c': 4})
        >>> parser.parse_args(['--conf', 'c.d=4'])
        Namespace(conf={'a': 4, 'c': {'d': 4}})
        >>> parser.parse_args(['--conf', 'c.d=4', '--conf', 'c=2'])
        Namespace(conf={'a': 4, 'c': 2})
        >>> parser.parse_args(['--conf', '{d: 5, e: 9}'])
        Namespace(conf={'d': 5, 'e': 9})

    """
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 default = None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        if default is None:
            default = {}

        if not isinstance(default, dict):
            raise TypeError('default must be str object: {}'
                            .format(type(default)))

        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            default=copy.deepcopy(default),
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_strings=None):
        # --{option} a.b=3 -> {'a': {'b': 3}}
        if '=' in values:
            indict = copy.deepcopy(getattr(namespace, self.dest, {}))
            key, value = values.split('=', maxsplit=1)
            if not value.strip() == '':
                value = yaml.load(value, Loader=yaml.Loader)
            if not isinstance(indict, dict):
                indict = {}

            keys = key.split('.')
            d = indict
            for idx, k in enumerate(keys):
                if idx == len(keys) - 1:
                    d[k] = value
                else:
                    v = d.setdefault(k, {})
                    if not isinstance(v, dict):
                        # Remove the existing value and recreates as empty dict
                        d[k] = {}
                    d = d[k]

            # Update the value
            setattr(namespace, self.dest, indict)
        else:
            setattr(namespace, self.dest, values)
            try:
                # At the first, try eval(), i.e. Python syntax dict.
                # e.g. --{option} "{'a': 3}" -> {'a': 3}
                # This is workaround for internal behaviour of configargparse.
                value = eval(values, {}, {})
                if not isinstance(value, dict):
                    raise ValueError
            except Exception:
                # and the second, try yaml.load
                value = yaml.load(values, Loader=yaml.Loader)
                if not isinstance(value, dict):
                    raise ValueError
            # Remove existing params, and overwrite
            setattr(namespace, self.dest, value)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

