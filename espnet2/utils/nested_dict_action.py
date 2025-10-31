"""Nested Dictionary Action for argparse.

This module provides a custom argparse Action class for parsing nested dictionary
configurations from command-line arguments. It supports various input formats including
dot-separated keys for nested structures, YAML syntax, and Python dict literals.
"""

import argparse
import copy

import yaml


class NestedDictAction(argparse.Action):
    """Action class to append items to dict object.

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

    _syntax = """Syntax:
  {op} <key>=<yaml-string>
  {op} <key>.<key2>=<yaml-string>
  {op} <python-dict>
  {op} <yaml-string>
e.g.
  {op} a=4
  {op} a.b={{c: true}}
  {op} {{"c": True}}
  {op} {{a: 34.5}}
"""

    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        default=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):
        """Initialize the NestedDictAction.

        Args:
            option_strings: List of option strings (e.g., ['-f', '--foo']).
            dest: Name of the attribute to store the parsed value.
            nargs: Number of arguments (not used, defaults to None).
            default: Default value if argument is not provided.
            choices: Container of allowable values (not used).
            required: Whether the argument is required.
            help: Description of the argument for help message.
            metavar: Name for the argument in usage messages.

        """
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            default=copy.deepcopy(default),
            type=None,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_strings=None):
        """Parse and process the argument value.

        Handles three types of input formats:

        1. Key-value format (with '='):
           - Simple: 'a=3' -> {'a': 3}
           - Nested: 'a.b.c=value' -> {'a': {'b': {'c': value}}}
           - YAML values: 'a={b: true}' -> {'a': {'b': True}}

        2. Python dict literal format:
           - Tries eval() first for Python dict syntax: "{'a': 3}"

        3. YAML dict format:
           - Falls back to YAML parsing if eval() fails

        Args:
            parser: The ArgumentParser instance.
            namespace: The Namespace object to store the result.
            values: The string value from the command line.
            option_strings: The option strings that triggered this action.

        Raises:
            argparse.ArgumentTypeError: If value cannot be interpreted as a dict.
            argparse.ArgumentError: If YAML parsing fails or value is not a dict.

        """
        # --{option} a.b=3 -> {'a': {'b': 3}}
        if "=" in values:
            indict = copy.deepcopy(getattr(namespace, self.dest, {}))
            key, value = values.split("=", maxsplit=1)
            if not value.strip() == "":
                value = yaml.load(value, Loader=yaml.Loader)
            if not isinstance(indict, dict):
                indict = {}

            keys = key.split(".")
            d = indict
            for idx, k in enumerate(keys):
                if idx == len(keys) - 1:
                    d[k] = value
                else:
                    if not isinstance(d.setdefault(k, {}), dict):
                        # Remove the existing value and recreates as empty dict
                        d[k] = {}
                    d = d[k]

            # Update the value
            setattr(namespace, self.dest, indict)
        else:
            try:
                # At the first, try eval(), i.e. Python syntax dict.
                # e.g. --{option} "{'a': 3}" -> {'a': 3}
                # This is workaround for internal behaviour of configargparse.
                value = eval(values, {}, {})
                if not isinstance(value, dict):
                    syntax = self._syntax.format(op=option_strings)
                    mes = f"must be interpreted as dict: but got {values}\n{syntax}"
                    raise argparse.ArgumentTypeError(self, mes)
            except Exception:
                # and the second, try yaml.load
                value = yaml.load(values, Loader=yaml.Loader)
                if not isinstance(value, dict):
                    syntax = self._syntax.format(op=option_strings)
                    mes = f"must be interpreted as dict: but got {values}\n{syntax}"
                    raise argparse.ArgumentError(self, mes)

            d = getattr(namespace, self.dest, None)
            if isinstance(d, dict):
                d.update(value)
            else:
                # Remove existing params, and overwrite
                setattr(namespace, self.dest, value)
