"""Clone command: copy a recipe to a local project directory."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_EXCLUDED_NAMES = {"__pycache__", "downloads", "downloads.tar.gz", "demo"}

_DESCRIPTION = """\
Copy an egs3 recipe to a new directory so you can customise it without
touching the original.

Copies the complete recipe package, excluding generated artefacts and caches.
The destination directory must not already exist.

If --project is omitted, the recipe name is used as the destination
(e.g. mini_an4/asr is cloned to ./mini_an4/asr/).
"""

_EPILOG = """\
examples:
  Clone using the recipe name as destination (./mini_an4/asr/):

    espnet3 clone mini_an4/asr
    cd mini_an4/asr
    python run.py --stages create_dataset train \\
        --training_config conf/training.yaml

  Clone into a custom directory:

    espnet3 clone mini_an4/asr --project my_asr
    cd my_asr
    python run.py --stages create_dataset train \\
        --training_config conf/training.yaml

  List all available recipes:

    espnet3 clone --list

  Run all stages end-to-end after cloning:

    espnet3 clone librispeech/asr --project ls_asr
    cd ls_asr
    python run.py --stages all \\
        --training_config conf/training.yaml \\
        --inference_config conf/inference.yaml \\
        --metrics_config conf/metrics.yaml

what gets cloned:
  The complete recipe package, including recipe-specific files and directories.
  Generated artefacts, caches, and hidden entries are excluded.
"""

_MISSING_RECIPE = """\
recipe argument is required.

  espnet3 clone <dataset>/<task> [--project <dir>]

examples:
  espnet3 clone mini_an4/asr                    # clones to ./mini_an4/asr/
  espnet3 clone mini_an4/asr --project my_asr   # clones to ./my_asr/

run 'espnet3 clone --list' to see all available recipes.
run 'espnet3 clone --help' for full usage.\
"""


def add_arguments(subparsers) -> None:
    """Register the clone subcommand onto an argparse subparsers object.

    Args:
        subparsers: The argparse ``_SubParsersAction`` returned by
            ``ArgumentParser.add_subparsers()``.

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
            " Example: mini_an4/asr, librispeech/asr"
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


def run(args) -> None:
    """Copy a complete egs3 recipe package into a new project directory.

    The recipe directory is the source of truth for its layout. Generated
    artefacts, caches, and hidden entries are excluded; all other files,
    including package markers and recipe-specific source, are copied.

    Args:
        args: Parsed CLI arguments containing a ``<dataset>/<task>`` recipe
            and an optional destination directory.

    Raises:
        FileNotFoundError: If the requested recipe is unavailable locally.
        FileExistsError: If the destination already exists.
        ValueError: If the recipe identifier has an invalid format.
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
        return

    if args.recipe is None:
        raise ValueError(_MISSING_RECIPE)

    from espnet3.cli.clone.resolver import resolve_recipe

    recipe_path = resolve_recipe(args.recipe)
    dest = Path(args.project).resolve() if args.project else Path.cwd() / args.recipe

    if dest.exists():
        raise FileExistsError(
            f"Destination already exists: {dest}\n"
            "Remove it or choose a different --project name."
        )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Cloning %s -> %s", args.recipe, dest)

    _copy_recipe(recipe_path, dest)
    _inject_corpus_system(dest, args.recipe)

    logger.info("Done.")
    logger.info("  cd %s", dest)
    logger.info("  python run.py --help")


def _copy_recipe(src: Path, dest: Path) -> None:
    """Copy a recipe package while excluding generated artefacts and caches."""
    shutil.copytree(src, dest, symlinks=False, ignore=_ignore_recipe_artifacts)


def _ignore_recipe_artifacts(_directory: str, names: list[str]) -> set[str]:
    """Return generated, cached, and hidden entries that clone must omit."""
    return {
        name
        for name in names
        if name in _EXCLUDED_NAMES or name.endswith(".pyc") or name.startswith(".")
    }


def _inject_corpus_system(dest: Path, recipe: str) -> None:
    from omegaconf import OmegaConf

    corpus_system = recipe.strip("/").replace("/", "_")

    pub_yaml = dest / "conf" / "publication.yaml"
    if pub_yaml.exists():
        conf = OmegaConf.load(pub_yaml)
        OmegaConf.update(
            conf,
            "upload_model.hf_repo",
            f"espnet/{corpus_system}_${{exp_tag}}",
            force_add=True,
        )
        OmegaConf.save(conf, pub_yaml)

    demo_yaml = dest / "conf" / "demo.yaml"
    if demo_yaml.exists():
        conf = OmegaConf.load(demo_yaml)
        OmegaConf.update(conf, "ui.title", f"{corpus_system} demo", force_add=True)
        OmegaConf.update(
            conf,
            "upload_demo.hf_repo",
            f"espnet/{corpus_system}_${{exp_tag}}",
            force_add=True,
        )
        OmegaConf.save(conf, demo_yaml)
