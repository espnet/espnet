import argparse
from pathlib import Path

import yaml


class ArgumentParser(argparse.ArgumentParser):
    """
        Simple implementation of ArgumentParser supporting config file.

    This class is originated from https://github.com/bw2/ConfigArgParse,
    but this class lacks some features that it has:

    - Not supporting multiple config files.
    - Automatically adding "--config" as an option.
    - Not supporting any formats other than YAML.
    - Not checking argument types.

    Attributes:
        config (str): Path to the YAML configuration file.

    Args:
        *args: Variable length argument list for the base ArgumentParser.
        **kwargs: Keyword arguments for the base ArgumentParser.

    Returns:
        Namespace: A Namespace object with parsed arguments.

    Raises:
        ArgumentError: If the configuration file does not exist or has invalid
            contents.

    Examples:
        parser = ArgumentParser()
        parser.add_argument('--example', help='An example argument.')
        args = parser.parse_known_args()

        If a YAML config file is specified with the `--config` argument,
        the parser will load the configurations from that file as well.

    Note:
        The parser does not check argument types, meaning any type of value
        can be set regardless of the expected argument type.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--config", help="Give config file in yaml format")

    def parse_known_args(self, args=None, namespace=None):
        """
        Parse the command line arguments, including those from a config file.

        This method extends the default `parse_known_args` to support loading
        configurations from a YAML file specified via the "--config" option.
        It first parses known arguments and then loads the configuration from the
        specified YAML file if it exists. The loaded configuration values are set
        as defaults for the argument parser.

        Args:
            args (list, optional): The list of command line arguments to parse.
                If None, uses `sys.argv[1:]`.
            namespace (argparse.Namespace, optional): An optional namespace
                to populate with the parsed arguments. If None, a new namespace
                is created.

        Returns:
            tuple: A tuple containing:
                - Namespace: An object containing the parsed arguments.
                - list: Any remaining unrecognized command line arguments.

        Raises:
            ArgumentError: If the specified config file does not exist or if
            the contents of the config file do not form a valid dictionary.
            RuntimeError: If any keys in the config file do not correspond
            to recognized arguments.

        Examples:
            >>> parser = ArgumentParser()
            >>> parser.add_argument('--foo', type=int)
            >>> parser.add_argument('--bar', type=str)
            >>> args, unknown = parser.parse_known_args(['--config', 'config.yaml'])

        Note:
            The method ignores the "--config" argument when loading from the
            config file, and does not enforce type checking for the values set
            from the config file.
        """
        # Once parsing for setting from "--config"
        _args, _ = super().parse_known_args(args, namespace)
        if _args.config is not None:
            if not Path(_args.config).exists():
                self.error(f"No such file: {_args.config}")

            with open(_args.config, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
            if not isinstance(d, dict):
                self.error("Config file has non dict value: {_args.config}")

            for key in d:
                for action in self._actions:
                    if key == action.dest:
                        break
                else:
                    self.error(f"unrecognized arguments: {key} (from {_args.config})")

            # NOTE(kamo): Ignore "--config" from a config file
            # NOTE(kamo): Unlike "configargparse", this module doesn't check type.
            #   i.e. We can set any type value regardless of argument type.
            self.set_defaults(**d)
        return super().parse_known_args(args, namespace)
