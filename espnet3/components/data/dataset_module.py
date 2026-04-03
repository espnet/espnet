"""Utilities for resolving and instantiating dataset modules.

This module standardizes how ESPnet3 resolves dataset references used in
configs such as:

.. code-block:: yaml

    dataset:
      train:
        - ref: mini_an4/asr
          kwargs:
            split: train

It supports:

1. Tag notation (``mini_an4/asr``) -> ``egs3.mini_an4.asr.dataset``.
2. Direct dotted module paths.
3. Local recipe datasets when ``ref`` is omitted.
"""

from __future__ import annotations

import sys
from importlib import import_module, util
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf


def _to_plain_dict(config: Any) -> dict[str, Any]:

    # Keep unresolved OmegaConf values as-is at this layer.
    if isinstance(config, DictConfig):
        return dict(OmegaConf.to_container(config, resolve=False))
    return dict(config)


# Dataset modules follow the convention: egs3.<tag>.dataset
# e.g. "mini_an4/asr" -> "egs3.mini_an4.asr.dataset"
_RECIPE_NAMESPACE = "egs3"
_DATASET_SUBMODULE = "dataset"


def _is_tag(ref: str) -> bool:
    return "/" in ref


def resolve_dataset_module_name(ref: str) -> str:
    """Resolve a tag-style ref to a Python module path.

    Args:
        ref: Tag-style dataset ref, typically ``<recipe>/<task>``.

    Returns:
        Fully qualified module path following
        ``egs3.<recipe>.<task>.dataset``.

    Raises:
        None.

    Notes:
        Hyphens are normalized to underscores to match Python module naming.

    Examples:
        >>> resolve_dataset_module_name("mini_an4/asr")
        'egs3.mini_an4.asr.dataset'
        >>> resolve_dataset_module_name("my-recipe/asr")
        'egs3.my_recipe.asr.dataset'
    """
    # Normalize each path segment to a Python module-safe name.
    normalized = ".".join(part.replace("-", "_") for part in ref.split("/"))
    return f"{_RECIPE_NAMESPACE}.{normalized}.{_DATASET_SUBMODULE}"


def _load_local_dataset_module(recipe_dir: str | Path | None):
    """Load ``recipe_dir/dataset`` as an importable local module.

    Args:
        recipe_dir: Recipe root directory. When ``None``, current working
            directory is used.

    Returns:
        Imported module object for ``dataset/__init__.py``.

    Raises:
        ModuleNotFoundError: If ``dataset/__init__.py`` does not exist.
        ImportError: If Python cannot construct an import spec for the file.

    Examples:
        >>> module = _load_local_dataset_module("egs3/mini_an4/asr")
        >>> hasattr(module, "Dataset")
        True
    """
    recipe_root = Path(recipe_dir) if recipe_dir is not None else Path.cwd()
    module_init = recipe_root / "dataset" / "__init__.py"
    if not module_init.is_file():
        raise ModuleNotFoundError(
            "No dataset ref provided and local dataset module not found at "
            f"{module_init}. Set `ref` or add recipe_dir/dataset/__init__.py."
        )

    # Use a unique synthetic module name to avoid collisions across recipes.
    sanitized = module_init.resolve().as_posix().replace("/", "_").replace(".", "_")
    module_name = f"_espnet3_local_dataset_{sanitized}"
    spec = util.spec_from_file_location(
        module_name,
        module_init,
        submodule_search_locations=[str(module_init.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load local dataset module: {module_init}")

    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_dataset_module(ref: str | None = None, recipe_dir: str | Path | None = None):
    """Load a dataset module from ref or local recipe path.

    Args:
        ref: Dataset reference. Supported forms:
            - ``None``: load local ``recipe_dir/dataset`` module.
            - tag form: ``mini_an4/asr``.
            - module path form: ``egs3.mini_an4.asr.dataset``.
        recipe_dir: Recipe root directory used for local module loading.

    Returns:
        Imported dataset module object.

    Raises:
        AssertionError: If both ``ref`` and ``recipe_dir`` are ``None``.
        ModuleNotFoundError: If target module cannot be found.
        ImportError: If import fails for other reasons.

    Notes:
        This function does not instantiate ``Dataset`` or ``DatasetBuilder``.
        It only resolves and imports the module.

    Examples:
        >>> module = load_dataset_module(ref="mini_an4/asr")
        >>> hasattr(module, "DatasetBuilder")
        True
        >>> local = load_dataset_module(ref=None, recipe_dir="egs3/mini_an4/asr")
        >>> hasattr(local, "Dataset")
        True
        >>> load_dataset_module(ref=None, recipe_dir=None)
        Traceback (most recent call last):
        ...
        AssertionError: recipe_dir must be set when ref is None.
    """
    if ref is None:
        assert recipe_dir is not None, "recipe_dir must be set when ref is None."

        # Local recipe mode: load <recipe_dir>/dataset/__init__.py directly.
        return _load_local_dataset_module(recipe_dir)
    if _is_tag(ref):

        # Tag mode: "mini_an4/asr" -> "egs3.mini_an4.asr.dataset"
        return import_module(resolve_dataset_module_name(ref))

    # Module path mode: import as-is.
    return import_module(ref)


def instantiate_dataset_reference(
    config: Mapping[str, Any] | DictConfig,
    recipe_dir: str | Path | None = None,
):
    """Instantiate a dataset class from a dataset entry config.

    Args:
        config: Dataset entry mapping. Expected keys are:
            - ``ref``: optional dataset reference.
            - ``kwargs``: optional constructor kwargs for ``Dataset``.
        recipe_dir: Recipe root used when resolving local modules.

    Returns:
        Instantiated dataset object from ``module.Dataset(**kwargs)``.

    Raises:
        AttributeError: If the target module does not expose ``Dataset``.
        ModuleNotFoundError: If dataset module resolution fails.
        TypeError: If ``kwargs`` is incompatible with dataset constructor.

    Notes:
        Only ``kwargs`` is forwarded to the dataset constructor.
        Top-level config fields other than ``kwargs`` are not passed through.

    Examples:
        >>> cfg = {"ref": "mini_an4/asr", "kwargs": {"split": "train"}}
        >>> ds = instantiate_dataset_reference(cfg, recipe_dir="egs3/mini_an4/asr")
        >>> hasattr(ds, "__len__")
        True

        Local dataset module loading (no ref):
        >>> cfg = {"kwargs": {"split": "test"}}
        >>> _ = instantiate_dataset_reference(cfg, recipe_dir="egs3/mini_an4/asr")
    """
    plain = _to_plain_dict(config)
    ref_value = plain.get("ref")

    # Empty/whitespace refs are treated as local dataset module usage.
    ref = (
        ref_value.strip() if isinstance(ref_value, str) and ref_value.strip() else None
    )
    raw_kwargs = plain.get("kwargs", {})

    # Only kwargs are forwarded into Dataset(...).
    kwargs = dict(raw_kwargs)
    module = load_dataset_module(ref=ref, recipe_dir=recipe_dir)
    dataset_cls = getattr(module, "Dataset")
    return dataset_cls(**kwargs)
