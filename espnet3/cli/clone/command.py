"""Clone command: copy a recipe to a local project directory."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_INCLUDE = {"conf", "src", "dataset", "run.py", "readme.md"}

_DESCRIPTION = """\
Copy an egs3 recipe to a new directory so you can customise it without
touching the original.

Clones conf/, src/, dataset/, run.py, readme.md, and path.sh.
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
  conf/       training / inference / metrics / publication / demo configs
  dataset/    dataset builder and config
  src/        inference helpers and app entry-points
  run.py      stage runner
  readme.md   recipe documentation
  path.sh     shell environment setup
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
    r"""Clone an egs3 recipe into a standalone project directory.

    Copies the recipe's ``conf/``, ``src/``, ``dataset/``, and top-level
    files (``run.py``, ``readme.md``, ``path.sh``) to a new directory so
    you can customise the experiment without touching the original recipe.

    The clone is a plain directory — not a Python package — so there is
    no ``__init__.py``.  Build artefacts and hidden entries are always
    excluded (see *Notes* for the full list).

    Once cloned, the project only needs ``espnet3`` to be installed; it
    does not need to live inside the ESPnet repository.

    Args:
        args: Parsed argument namespace produced by ``add_arguments``.
            Expected attributes:

            * ``recipe`` (``str``) — recipe identifier in
              ``"<dataset>/<task>"`` format, e.g. ``"mini_an4/asr"``.
            * ``project`` (``str``) — path of the directory to create.

    Raises:
        FileNotFoundError: If *recipe* is not found in ``egs3/``.  The
            error message lists all currently available recipes.
        FileExistsError: If the destination directory already exists.
            Remove it or choose a different ``--project`` name.
        ValueError: If *recipe* is not in ``"<dataset>/<task>"`` format.

    Notes:
        **What gets cloned**

        +--------------+--------------------------------------------------+
        | Path         | Contents                                         |
        +==============+==================================================+
        | ``conf/``    | YAML configs for training, inference, metrics,   |
        |              | publication, and demo stages.                    |
        +--------------+--------------------------------------------------+
        | ``dataset/`` | Dataset builder, dataset class, config YAML.     |
        +--------------+--------------------------------------------------+
        | ``src/``     | Inference helpers and app entry-points.          |
        +--------------+--------------------------------------------------+
        | ``run.py``   | Stage runner; imports shared logic from          |
        |              | ``egs3.TEMPLATE.<task>.run``.                    |
        +--------------+--------------------------------------------------+
        | ``readme.md``| Recipe-level documentation.                      |
        +--------------+--------------------------------------------------+
        | ``path.sh``  | Shell environment setup.                         |
        +--------------+--------------------------------------------------+

        **What is excluded**

        ``__init__.py``, ``__pycache__/``, ``downloads/``,
        ``downloads.tar.gz``, ``demo/``, ``empty.py``, and any entry
        whose name begins with ``"."`` (e.g. ``.git``, ``.agents``).

    Examples:
        **Basic clone and first run**

        Clone ``mini_an4/asr`` and immediately start training::

            $ espnet3 clone mini_an4/asr --project my_asr
            Cloning mini_an4/asr -> /home/user/my_asr
            Done.
              cd my_asr
              python run.py --help

            $ cd my_asr
            $ python run.py \\
                --stages create_dataset train \\
                --training_config conf/training.yaml

        **Resulting directory layout**

        The cloned directory contains everything needed to run and
        customise the experiment::

            my_asr/
            ├── conf/
            │   ├── training.yaml
            │   ├── inference.yaml
            │   ├── metrics.yaml
            │   ├── publication.yaml
            │   └── demo.yaml
            ├── dataset/
            │   ├── builder.py
            │   ├── dataset.py
            │   └── config.yaml
            ├── src/
            │   ├── app.py
            │   ├── inference.py
            │   └── tokenizer.py
            ├── run.py
            ├── readme.md
            └── path.sh

        **Running all stages end-to-end**

        After cloning, pass each config explicitly to cover every stage
        from data preparation through model publishing::

            $ espnet3 clone librispeech/asr --project ls_asr
            $ cd ls_asr
            $ python run.py \\
                --stages all \\
                --training_config conf/training.yaml \\
                --inference_config conf/inference.yaml \\
                --metrics_config conf/metrics.yaml \\
                --publication_config conf/publication.yaml \\
                --demo_config conf/demo.yaml

        **Overriding a single config value**

        Hydra override syntax works directly on the command line, so
        you can change e.g. the encoder type without editing any file::

            $ python run.py \\
                --stages train \\
                --training_config conf/training.yaml \\
                encoder=conformer \\
                encoder_conf.num_blocks=12

        **Dry-run before committing to a long job**

        Use ``--dry_run`` to print every stage that *would* run without
        actually executing anything::

            $ python run.py \\
                --stages all \\
                --training_config conf/training.yaml \\
                --dry_run

        **Error: destination already exists**

        The command refuses to overwrite an existing directory::

            $ espnet3 clone mini_an4/asr --project existing_dir
            error: Destination already exists: /home/user/existing_dir
            Remove it or choose a different --project name.

        **Error: unknown recipe (shows available list)**

        If the recipe name is wrong, all known recipes are listed::

            $ espnet3 clone bad_name/asr --project proj
            error: Recipe 'bad_name/asr' not found in .../egs3.
            Available recipes:
              falar/asr
              mini_an4/asr
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
    """Copy a recipe tree into ``dest``.

    Directory contents are copied with ``shutil.copytree()``. Nested symlinks
    must therefore be filtered through the copytree ignore callback rather
    than only at this top level.
    """
    dest.mkdir(parents=True)
    for item in _INCLUDE:
        source = src / item
        if not source.exists():
            continue
        if source.is_symlink():
            continue
        if source.is_dir():
            shutil.copytree(
                source,
                dest / item,
                ignore=_ignore_copytree_entries,
            )
        else:
            shutil.copy2(source, dest / item)


def _ignore_copytree_entries(dirpath: str, names: list[str]) -> set[str]:
    """Skip nested symlinks when cloning a directory tree.

    ``_copy_recipe()`` only sees top-level entries such as ``conf/``. Files
    like ``conf/training.yaml`` are encountered later inside
    ``shutil.copytree()``, so symlinks under cloned directories must be
    ignored here.
    """
    base = Path(dirpath)
    ignored = set(shutil.ignore_patterns("__pycache__", "*.pyc")(dirpath, names))
    ignored.update(name for name in names if (base / name).is_symlink())
    return ignored


def _inject_corpus_system(dest: Path, recipe: str) -> None:
    corpus_system = recipe.strip("/").replace("/", "_")

    pub_yaml = dest / "conf" / "publication.yaml"
    if pub_yaml.exists():
        with pub_yaml.open("a", encoding="utf-8") as f:
            f.write(
                f"\n# Corpus/system identity injected by espnet3 clone.\n"
                f"upload_model:\n"
                f"  hf_repo: espnet/{corpus_system}_${{exp_tag}}\n"
            )

    demo_yaml = dest / "conf" / "demo.yaml"
    if demo_yaml.exists():
        with demo_yaml.open("a", encoding="utf-8") as f:
            f.write(
                f"\n# Corpus/system identity injected by espnet3 clone.\n"
                f"ui:\n"
                f"  title: {corpus_system} demo\n"
                f"upload_demo:\n"
                f"  hf_repo: espnet/{corpus_system}_${{exp_tag}}\n"
            )
