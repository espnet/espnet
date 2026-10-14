"""Clone command: copy a recipe to a local project directory."""

from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
import shutil
from pathlib import Path

_IGNORED_ANYWHERE = ("*.pyc", "__pycache__", ".pytest_cache")
# These mirror the egs*/<dataset>/<task>/ rules in .gitignore. `data` is exact:
# `dataset/` is recipe source and must be copied.
_IGNORED_RECIPE_ROOT = (
    "data",
    "manifest*",
    "db",
    "downloads",
    "dump",
    "enhan",
    "exp",
    "fbank",
    "mfcc",
    "stft",
    "tensorboard",
    "wav*",
    "score*",
    "nltk*",
    ".cache*",
    "pretrained_models*",
)

_DESCRIPTION = """\
Copy an egs3 recipe to a new directory so you can customise it without
touching the original.

Copies the complete recipe package, excluding generated artefacts and caches.
The destination directory must not already exist.

If --project is omitted, the recipe name is used as the destination
(e.g. mini_an4/esp2_asr is cloned to ./mini_an4/esp2_asr/).
"""

_EPILOG = """\
examples:
  Clone using the recipe name as destination (./mini_an4/esp2_asr/):

    espnet clone mini_an4/esp2_asr
    cd mini_an4/esp2_asr
    python run.py --stages create_dataset train \\
        --training_config conf/training.yaml

  Clone into a custom directory:

    espnet clone mini_an4/esp2_asr --project my_asr
    cd my_asr
    python run.py --stages create_dataset train \\
        --training_config conf/training.yaml

  List all available recipes:

    espnet clone --list

  Run all stages end-to-end after cloning:

    espnet clone librispeech_100/esp2_asr --project ls_asr
    cd ls_asr
    python run.py --stages all \\
        --training_config conf/training.yaml \\
        --inference_config conf/inference.yaml \\
        --metrics_config conf/metrics.yaml

what gets cloned:
  The complete recipe package, including recipe-specific files and directories.
  Generated artefacts and caches ignored by the recipe policy are excluded.
"""

_MISSING_RECIPE = """\
recipe argument is required.

  espnet clone <dataset>/<task> [--project <dir>]

examples:
  espnet clone mini_an4/esp2_asr                    # clones to ./mini_an4/esp2_asr/
  espnet clone mini_an4/esp2_asr --project my_asr   # clones to ./my_asr/

run 'espnet clone --list' to see all available recipes.
run 'espnet clone --help' for full usage.\
"""


def add_arguments(subparsers) -> argparse.ArgumentParser:
    """Register the clone subcommand onto an argparse subparsers object.

    Args:
        subparsers: The argparse subparser collection returned by
            ``ArgumentParser.add_subparsers()``.

    Returns:
        The parser created for the ``clone`` subcommand.

    Examples:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> sub = parser.add_subparsers()
        >>> add_arguments(sub)
    """
    parser = subparsers.add_parser(
        "clone",
        help="Clone an egs3 recipe into a local project directory.",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "recipe",
        nargs="?",
        default=None,
        metavar="<dataset>/<task>",
        help=(
            "Recipe to clone in <dataset>/<task> format."
            " Example: mini_an4/esp2_asr, librispeech_100/esp2_asr"
        ),
    )
    parser.add_argument(
        "--project",
        default=None,
        metavar="DIR",
        help=(
            "Destination directory to create (must not already exist)."
            " Defaults to ./<recipe> when omitted."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List all available recipes and exit.",
    )
    parser.set_defaults(func=run)
    return parser


def run(args) -> int:
    """Copy a complete egs3 recipe package into a new project directory.

    The recipe directory is the source of truth for its layout. Generated
    artefacts and caches ignored by the recipe policy are excluded; all other
    files, including package markers and recipe-specific source, are copied.

    Args:
        args: Parsed CLI arguments containing a ``<dataset>/<task>`` recipe
            and an optional destination directory.

    Raises:
        FileNotFoundError: If the requested recipe is unavailable locally.
        FileExistsError: If the destination already exists.
        ValueError: If the recipe identifier has an invalid format.

    Examples:
        ```python
        import argparse

        run(argparse.Namespace(
            list=False, recipe="mini_an4/esp2_asr", project="my_asr"))
        ```
    """
    if args.list:
        from espnet3.cli.clone.resolver import list_recipes

        recipes = list_recipes()
        if not recipes:
            print("No recipes available.")
        else:
            print("Available recipes:")
            for recipe in recipes:
                print(f"  {recipe}")
        return 0

    if args.recipe is None:
        raise ValueError(_MISSING_RECIPE)

    from espnet3.cli.clone.resolver import resolve_recipe

    recipe_path = resolve_recipe(args.recipe)
    recipe_name = args.recipe.strip("/")
    dest = Path(args.project).resolve() if args.project else Path.cwd() / recipe_name

    if dest.exists():
        raise FileExistsError(
            f"Destination already exists: {dest}\n"
            "Remove it or choose a different --project name."
        )

    print(f"Cloning {args.recipe} -> {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    _copy_recipe(recipe_path, dest)

    print("Done.")
    print(f"  cd {dest}")
    print("  python run.py --help")
    return 0


def _copy_recipe(src: Path, dest: Path) -> None:
    """Copy a recipe package while excluding generated artefacts and caches."""
    source = Path(src).resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        return _ignore_recipe_artifacts(directory, names, source)

    shutil.copytree(source, dest, symlinks=False, ignore=ignore)


def _ignore_recipe_artifacts(
    directory: str, names: list[str], recipe_root: Path
) -> set[str]:
    """Return generated and cached entries ignored by the recipe policy."""
    at_recipe_root = Path(directory).resolve() == recipe_root
    return {
        name
        for name in names
        if any(fnmatchcase(name, pattern) for pattern in _IGNORED_ANYWHERE)
        or (
            at_recipe_root
            and any(fnmatchcase(name, pattern) for pattern in _IGNORED_RECIPE_ROOT)
        )
    }
