"""Configuration file argument parsing utilities.

This module provides an ArgumentParser that extends the standard argparse module
to support YAML configuration files. It automatically adds a "--config" option
and allows users to specify default arguments via YAML files.
"""

import argparse
from pathlib import Path

import yaml


class ArgumentParser(argparse.ArgumentParser):
    """Simple implementation of ArgumentParser supporting config file.

    This class is originated from https://github.com/bw2/ConfigArgParse,
    but this class is lack of some features that it has.

    - Not supporting multiple config files
    - Automatically adding "--config" as an option.
    - Not supporting any formats other than yaml
    - Not checking argument type

    """

    def __init__(self, *args, **kwargs):
        """Initialize the ArgumentParser with config file support.

        Initializes the ArgumentParser and automatically adds a "--config" argument
        to allow users to specify a YAML configuration file containing default values.

        Args:
            *args: Positional arguments passed to argparse.ArgumentParser.
            **kwargs: Keyword arguments passed to argparse.ArgumentParser.

        """
        super().__init__(*args, **kwargs)
        self.add_argument("--config", help="Give config file in yaml format")

    def parse_known_args(self, args=None, namespace=None):
        """Parse known arguments and load defaults from config file if specified.

        This method overrides the standard parse_known_args to support loading
        default argument values from a YAML configuration file. If a "--config"
        argument is provided, it loads the YAML file and sets those values as
        defaults before parsing the remaining arguments.

        Args:
            args: List of strings to parse. If None, uses sys.argv.
            namespace: The Namespace object to take the attributes. If None,
                a new empty Namespace will be created.

        Returns:
            tuple: A tuple of (namespace, remaining_args) where namespace contains
                the parsed arguments and remaining_args contains any unrecognized
                arguments.

        Raises:
            SystemExit: If the config file is not found or contains invalid values.

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
