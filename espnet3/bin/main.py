"""ESPnet3 command-line entry point."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ``espnet3`` command.

    Args:
        argv: Command-line arguments without the program name. Defaults to
            ``sys.argv[1:]``.

    Returns:
        The process exit code. Expected user errors (missing or existing
        paths, invalid recipe names) are reported without a traceback and
        return 1.

    Examples:
        >>> main(["clone", "--list"])  # doctest: +SKIP
        0
    """
    from espnet3.bin.clone.command import add_arguments as add_clone_arguments

    parser = argparse.ArgumentParser(
        prog="espnet3", description="ESPnet3 command-line interface."
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True
    add_clone_arguments(subparsers)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"espnet3: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
