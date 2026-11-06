"""Clone command: copy a recipe to a local project directory."""

from __future__ import annotations

import argparse
import shutil
from fnmatch import fnmatchcase
from pathlib import Path

_INCLUDED_DIRS = {"src", "conf", "dataset"}
_INCLUDED_FILES = ("*.py", "README.md", "readme.md", "requirements*.txt")

_DESCRIPTION = """\
Copy an egs3 recipe to a new directory so you can customise it without
touching the original.

Copies src/, conf/, dataset/, root Python files, README files and requirements.
Hidden entries, caches and directory symlinks are excluded.
The destination directory must not already exist.

If --project is omitted, the recipe name is used as the destination
(e.g. mini_an4/esp2_asr is cloned to ./mini_an4/esp2_asr/).
"""

_EPILOG = """\
examples:
  Clone using the recipe name as destination (./mini_an4/esp2_asr/):

    espnet3 clone mini_an4/esp2_asr
    cd mini_an4/esp2_asr
    python run.py --stages create_dataset train \\
        --training_config conf/training.yaml

  Clone into a custom directory:

    espnet3 clone mini_an4/esp2_asr --project my_asr
    cd my_asr
    python run.py --stages create_dataset train \\
        --training_config conf/training.yaml

  List all available recipes:

    espnet3 clone --list

  Run all stages end-to-end after cloning:

    espnet3 clone librispeech_100/esp2_asr --project ls_asr
    cd ls_asr
    python run.py --stages all \\
        --training_config conf/training.yaml \\
        --inference_config conf/inference.yaml \\
        --metrics_config conf/metrics.yaml

what gets cloned:
  src/, conf/, dataset/, root *.py, README.md, readme.md and requirements*.txt.
  Hidden entries, caches and directory symlinks are excluded.
  File symlinks are copied as regular files. path.sh is not copied.
  Run with Python from an environment where ESPnet is installed.
"""

_MISSING_RECIPE = """\
recipe argument is required.

  espnet3 clone <dataset>/<task> [--project <dir>]

examples:
  espnet3 clone mini_an4/esp2_asr                    # clones to ./mini_an4/esp2_asr/
  espnet3 clone mini_an4/esp2_asr --project my_asr   # clones to ./my_asr/

run 'espnet3 clone --list' to see all available recipes.
run 'espnet3 clone --help' for full usage.\
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
    """Copy an egs3 recipe's source and configuration into a new project directory.

    Copy src/, conf/, dataset/, root Python files, README files and
    requirements files. Skip hidden entries, caches and directory symlinks.
    File symlinks are copied as regular files; path.sh is not copied.
    Use Python from an environment with ESPnet installed.

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
        from espnet3.bin.clone.resolver import list_recipes

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

    from espnet3.bin.clone.resolver import resolve_recipe

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
    if (dest / "conf" / "publication.yaml").is_file():
        print(
            "Warning: the default Hugging Face upload target uses the "
            "espnet organization. To publish under your own account, "
            "change upload_model.hf_repo in conf/publication.yaml."
        )
    return 0


def _copy_recipe(src: Path, dest: Path) -> None:
    """Copy recipe source, dereferencing file links and skipping directory links."""
    source = Path(src).resolve()

    dest.mkdir(parents=True)
    for entry in source.iterdir():
        if entry.name in _INCLUDED_DIRS or (
            entry.is_file()
            and any(fnmatchcase(entry.name, pattern) for pattern in _INCLUDED_FILES)
        ):
            _copy_recipe_entry(entry, dest / entry.name)
    _strip_recipe_module_prefix(dest, f"egs3.{source.parent.name}.{source.name}.")


def _strip_recipe_module_prefix(root: Path, prefix: str) -> None:
    """Rewrite a recipe's dotted module paths so they are relative to ``root``.

    Recipes import their own modules by absolute path, for example
    ``egs3.mini_an4.esp2_asr.dataset.builder``. A cloned copy lives outside
    ``egs3``, where that path would import the installed recipe instead of the
    copy, so the recipe prefix is removed (``dataset.builder``). The cloned
    ``run.py`` is executed with ``root`` on Python's module search path.

    Args:
        root: Directory of the cloned recipe.
        prefix: Dotted prefix to remove, including the trailing dot, such as
            ``"egs3.mini_an4.esp2_asr."``.

    Notes:
        Every regular file is checked as bytes. Binary files (containing a NUL
        byte) and files without the prefix are left untouched.

    Examples:
        >>> _strip_recipe_module_prefix(
        ...     Path("my_asr"), "egs3.mini_an4.esp2_asr."
        ... )  # doctest: +SKIP
    """
    needle = prefix.encode()
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if needle not in data or b"\0" in data:
            continue
        path.write_bytes(data.replace(needle, b""))


def _copy_recipe_entry(source: Path, destination: Path) -> None:
    """Copy a selected entry, skipping hidden entries, caches and directory links."""
    if source.name.startswith(".") or source.name == "__pycache__":
        return
    if source.suffix == ".pyc":
        return
    if source.is_dir():
        if source.is_symlink():
            print(f"Skipping directory symlink: {source}")
            return
        destination.mkdir()
        for child in source.iterdir():
            _copy_recipe_entry(child, destination / child.name)
    else:
        shutil.copy2(source, destination)
