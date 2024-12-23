import argparse
import copy

import yaml


class NestedDictAction(argparse.Action):
    """
        Action class to append items to a dictionary object.

    This class extends the `argparse.Action` to allow the user to pass
    configuration options in a nested dictionary format via command-line
    arguments. It supports both key-value pairs and YAML strings.

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

    Attributes:
        _syntax (str): Syntax for using the action in command-line arguments.

    Args:
        option_strings (list): The option strings for the action.
        dest (str): The name of the attribute to be added to the namespace.
        nargs (int or str, optional): The number of command-line arguments
            that should be consumed.
        default (dict, optional): The default value for the destination.
        choices (list, optional): The allowable choices for the argument.
        required (bool, optional): Whether or not the action is required.
        help (str, optional): The help text for the argument.
        metavar (str, optional): A name for the argument in usage messages.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be interpreted as a
            dictionary.
        argparse.ArgumentError: If the value cannot be interpreted as a
            dictionary when using YAML.

    Note:
        The values can be specified in the following formats:
          - {key}={value}
          - {key}.{subkey}={value}
          - {key: value, ...} (Python dict syntax)
          - {key: value, ...} (YAML syntax)

    Todo:
        - Enhance error messages for better user guidance.
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
