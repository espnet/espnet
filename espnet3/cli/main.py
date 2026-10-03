"""ESPnet3 CLI entry point."""

from __future__ import annotations

import argparse


def main() -> None:
    """Run the espnet3 command-line interface.

    Dispatches to subcommands registered under ``espnet3/cli/``.

    Examples:
        >>> # espnet3 clone mini_an4/asr --project my_project
        >>> # espnet3 --help
    """
    parser = argparse.ArgumentParser(
        prog="espnet3",
        description="ESPnet3 command-line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    from espnet3.cli.clone.command import add_arguments as add_clone

    add_clone(subparsers)

    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
