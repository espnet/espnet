# Copyright (C) 2012 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause
from argparse import RawDescriptionHelpFormatter
from os.path import lexists

from conda.base.context import context, determine_target_prefix
from conda.cli import common
from conda.cli.conda_argparse import add_parser_json, add_parser_prefix
from conda.core.prefix_data import PrefixData
from conda.exceptions import EnvironmentLocationNotFound

var_description = """
Interact with environment variables associated with Conda environments
"""

var_example = """
examples:
    conda env config vars list -n my_env
    conda env config vars set MY_VAR=something OTHER_THING=ohhhhya
    conda env config vars unset MY_VAR
"""

list_description = """
List environment variables for a conda environment
"""

list_example = """
examples:
    conda env config vars list -n my_env
"""

set_description = """
Set environment variables for a conda environment
"""

set_example = """
example:
    conda env config vars set MY_VAR=weee
"""

unset_description = """
Unset environment variables for a conda environment
"""

unset_example = """
example:
    conda env config vars unset MY_VAR
"""


def configure_parser(sub_parsers):
    var_parser = sub_parsers.add_parser(
        "vars",
        formatter_class=RawDescriptionHelpFormatter,
        description=var_description,
        help=var_description,
        epilog=var_example,
    )
    var_subparser = var_parser.add_subparsers()

    list_parser = var_subparser.add_parser(
        "list",
        formatter_class=RawDescriptionHelpFormatter,
        description=list_description,
        help=list_description,
        epilog=list_example,
    )
    add_parser_prefix(list_parser)
    add_parser_json(list_parser)
    list_parser.set_defaults(func=".main_vars.execute_list")

    set_parser = var_subparser.add_parser(
        "set",
        formatter_class=RawDescriptionHelpFormatter,
        description=set_description,
        help=set_description,
        epilog=set_example,
    )
    set_parser.add_argument(
        "vars",
        action="store",
        nargs="*",
        help="Environment variables to set in the form <KEY>=<VALUE> separated by spaces",
    )
    add_parser_prefix(set_parser)
    set_parser.set_defaults(func=".main_vars.execute_set")

    unset_parser = var_subparser.add_parser(
        "unset",
        formatter_class=RawDescriptionHelpFormatter,
        description=unset_description,
        help=unset_description,
        epilog=unset_example,
    )
    unset_parser.add_argument(
        "vars",
        action="store",
        nargs="*",
        help="Environment variables to unset in the form <KEY> separated by spaces",
    )
    add_parser_prefix(unset_parser)
    unset_parser.set_defaults(func=".main_vars.execute_unset")


def execute_list(args, parser):
    prefix = determine_target_prefix(context, args)
    if not lexists(prefix):
        raise EnvironmentLocationNotFound(prefix)

    pd = PrefixData(prefix)

    env_vars = pd.get_environment_env_vars()
    if args.json:
        common.stdout_json(env_vars)
    else:
        for k, v in env_vars.items():
            print(f"{k} = {v}")


def execute_set(args, parser):
    prefix = determine_target_prefix(context, args)
    pd = PrefixData(prefix)
    if not lexists(prefix):
        raise EnvironmentLocationNotFound(prefix)

    env_vars_to_add = {}
    for v in args.vars:
        var_def = v.split("=")
        env_vars_to_add[var_def[0].strip()] = "=".join(var_def[1:]).strip()
    pd.set_environment_env_vars(env_vars_to_add)
    if prefix == context.active_prefix:
        print("To make your changes take effect please reactivate your environment")


def execute_unset(args, parser):
    prefix = determine_target_prefix(context, args)
    pd = PrefixData(prefix)
    if not lexists(prefix):
        raise EnvironmentLocationNotFound(prefix)

    vars_to_unset = [_.strip() for _ in args.vars]
    pd.unset_environment_env_vars(vars_to_unset)
    if prefix == context.active_prefix:
        print("To make your changes take effect please reactivate your environment")
