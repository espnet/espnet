"""Utilities for resolving and instantiating dataset modules.

This module standardizes how ESPnet3 resolves dataset sources used in configs
such as:

.. code-block:: yaml

    dataset:
      train:
        - data_src: mini_an4/asr
          data_src_args:
            split: train

It supports:

1. Tag notation (``mini_an4/asr``) -> ``egs3.mini_an4.asr.dataset``.
2. Direct dotted module paths.
3. Local recipe datasets when ``data_src`` is omitted.
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
_DATA_SRC_KEY = "data_src"
_DATA_SRC_ARGS_KEY = "data_src_args"


def _is_tag(ref: str) -> bool:
    return "/" in ref


def resolve_dataset_module_name(ref: str) -> str:
    """Resolve a tag-style ref to a Python module path.

    Args:
        ref: Tag-style dataset ref, typically ``<recipe>/<task>``.

    Returns:
        Fully qualified module path following
        ``egs3.<recipe>.<task>.dataset``.

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
            "No dataset source provided and local dataset module not found at "
            f"{module_init}. Set `data_src` or add recipe_dir/dataset/__init__.py."
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


def load_dataset_module(
    data_src: str | None = None, recipe_dir: str | Path | None = None
):
    """Load a dataset module from a dataset source or local recipe path.

    Args:
        data_src: Dataset source reference. Supported forms:
            - ``None``: load local ``recipe_dir/dataset`` module.
            - tag form: ``mini_an4/asr``.
            - module path form: ``egs3.mini_an4.asr.dataset``.
        recipe_dir: Recipe root directory used for local module loading.

    Returns:
        Imported dataset module object.

    Raises:
        AssertionError: If both ``data_src`` and ``recipe_dir`` are ``None``.
        ModuleNotFoundError: If target module cannot be found.
        ImportError: If import fails for other reasons.

    Notes:
        This function does not instantiate ``Dataset`` or ``DatasetBuilder``.
        It only resolves and imports the module.

    Examples:
        >>> module = load_dataset_module(data_src="mini_an4/asr")
        >>> hasattr(module, "DatasetBuilder")
        True
        >>> local = load_dataset_module(
        ...     data_src=None, recipe_dir="egs3/mini_an4/asr"
        ... )
        >>> hasattr(local, "Dataset")
        True
        >>> load_dataset_module(data_src=None, recipe_dir=None)
        Traceback (most recent call last):
        ...
        AssertionError: recipe_dir must be set when data_src is None.
    """
    if data_src is None:
        assert recipe_dir is not None, "recipe_dir must be set when data_src is None."

        # Local recipe mode: load <recipe_dir>/dataset/__init__.py directly.
        return _load_local_dataset_module(recipe_dir)
    if _is_tag(data_src):

        # Tag mode: "mini_an4/asr" -> "egs3.mini_an4.asr.dataset"
        return import_module(resolve_dataset_module_name(data_src))

    # Module path mode: import as-is.
    return import_module(data_src)


def parse_dataset_reference_config(
    config: Mapping[str, Any] | DictConfig,
):
    """Extract dataset source fields from one dataset config entry.

    Args:
        config: Dataset entry mapping. Expected keys are:
            - ``data_src``: optional dataset source reference.
            - ``data_src_args``: optional constructor kwargs for ``Dataset``.

    Returns:
        Tuple of normalized ``(data_src, data_src_args)``.

    Raises:
        TypeError: If ``data_src`` is neither a string nor ``None``.
        ValueError: If ``data_src_args`` is incompatible with ``dict(...)``.

    Notes:
        Only ``data_src_args`` is forwarded to the dataset constructor.
        Top-level config fields such as ``name`` and ``transform`` are ignored
        here.

    Examples:
        >>> parse_dataset_reference_config(
        ...     {"data_src": "mini_an4/asr", "data_src_args": {"split": "train"}}
        ... )
        ('mini_an4/asr', {'split': 'train'})
        >>> parse_dataset_reference_config({"data_src_args": {"split": "test"}})
        (None, {'split': 'test'})
    """
    plain = _to_plain_dict(config)
    data_src = plain.get(_DATA_SRC_KEY)
    if data_src is not None:
        if not isinstance(data_src, str):
            raise TypeError("`data_src` must be a string or None.")
        data_src = data_src.strip() or None
    data_src_args = dict(plain.get(_DATA_SRC_ARGS_KEY, {}))
    return data_src, data_src_args


def instantiate_dataset_reference(
    config: Mapping[str, Any] | DictConfig,
    recipe_dir: str | Path | None = None,
):
    """Instantiate a dataset class from a dataset entry config.

    Args:
        config: Dataset entry mapping. Expected keys are:
            - ``data_src``: optional dataset source reference.
            - ``data_src_args``: optional constructor kwargs for ``Dataset``.
        recipe_dir: Recipe root used when resolving local modules.

    Returns:
        Instantiated dataset object from ``module.Dataset(**data_src_args)``.

    Raises:
        AttributeError: If the target module does not expose ``Dataset``.
        ModuleNotFoundError: If dataset module resolution fails.
        TypeError: If ``data_src_args`` is incompatible with dataset
            constructor.

    Notes:
        Only ``data_src_args`` is forwarded to the dataset constructor.
        Top-level config fields other than ``data_src_args`` are not passed
        through.

    Examples:
        >>> cfg = {"data_src": "mini_an4/asr", "data_src_args": {"split": "train"}}
        >>> ds = instantiate_dataset_reference(cfg, recipe_dir="egs3/mini_an4/asr")
        >>> hasattr(ds, "__len__")
        True

        Local dataset module loading (no data_src):
        >>> cfg = {"data_src_args": {"split": "test"}}
        >>> _ = instantiate_dataset_reference(cfg, recipe_dir="egs3/mini_an4/asr")
    """
    data_src, data_src_args = parse_dataset_reference_config(config)
    module = load_dataset_module(data_src=data_src, recipe_dir=recipe_dir)
    dataset_cls = getattr(module, "Dataset")
    return dataset_cls(**data_src_args)
