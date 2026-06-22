"""Recipe resolver for the clone command."""

from __future__ import annotations

from pathlib import Path


def _get_egs3_root() -> Path:
    import egs3

    return Path(egs3.__file__).parent


def resolve_recipe(recipe: str) -> Path:
    """Resolve a recipe identifier to a local path in egs3/.

    Args:
        recipe: Recipe identifier in "<dataset>/<task>" format,
            e.g. ``"mini_an4/asr"``.

    Returns:
        Absolute path to the recipe directory inside egs3/.

    Raises:
        ValueError: If the recipe identifier is not in
            ``<dataset>/<task>`` format.
        FileNotFoundError: If the recipe is not found locally.

    Notes:
        Currently only resolves from the local egs3/ directory.
        GitHub fallback is planned for a future release.

    Examples:
        >>> from pathlib import Path
        >>> path = resolve_recipe("mini_an4/asr")
        >>> path.name
        'asr'
        >>> path.parent.name
        'mini_an4'
    """
    parts = recipe.strip("/").split("/")
    if len(parts) != 2:
        raise ValueError(
            f"Recipe must be in '<dataset>/<task>' format, got: {recipe!r}"
        )
    dataset, task = parts
    egs3 = _get_egs3_root()
    recipe_path = egs3 / dataset / task
    if not recipe_path.exists():
        available = _list_available(egs3)
        hint = "\n  ".join(available) if available else "(none found)"
        raise FileNotFoundError(
            f"Recipe {recipe!r} not found in {egs3}.\n" f"Available recipes:\n  {hint}"
        )
    return recipe_path


def list_recipes() -> list[str]:
    """List all available recipe identifiers in egs3/.

    Returns:
        Sorted list of ``"<dataset>/<task>"`` strings.

    Examples:
        >>> recipes = list_recipes()
        >>> all("/" in r for r in recipes)
        True
    """
    return _list_available(_get_egs3_root())


def _list_available(egs3: Path) -> list[str]:
    _SKIP = {"TEMPLATE", "__pycache__"}
    recipes = []
    for dataset_dir in sorted(egs3.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_dir.name in _SKIP or dataset_dir.name.startswith("."):
            continue
        for task_dir in sorted(dataset_dir.iterdir()):
            if task_dir.is_dir() and not task_dir.name.startswith((".", "_")):
                recipes.append(f"{dataset_dir.name}/{task_dir.name}")
    return recipes
