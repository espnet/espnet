"""Utilities for resolving and instantiating dataset modules.

This module standardizes how ESPnet3 resolves dataset sources used in configs
such as:

.. code-block:: yaml

    dataset:
      train:
        - data_src: mini_an4/esp2_asr
          data_src_args:
            split: train

It supports:

1. Tag notation (``mini_an4/esp2_asr``) -> ``egs3.mini_an4.esp2_asr.dataset``.
2. Direct dotted module paths, whose ``Dataset`` attribute is instantiated.
3. Fully qualified dataset class paths, such as
   ``egs3.libritts.f5tts.dataset.librispeech_pc.LibriSpeechPCDataset``.
4. Local recipe datasets when ``data_src`` is omitted.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
from importlib import import_module, util
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from hydra.utils import get_object
from omegaconf import DictConfig, OmegaConf

from espnet3.components.data.base_dataset import BaseDataset


def _to_plain_dict(config: Any) -> dict[str, Any]:

    # Keep unresolved OmegaConf values as-is at this layer.
    if isinstance(config, DictConfig):
        return dict(OmegaConf.to_container(config, resolve=False))
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)
    return dict(config)


# Dataset modules follow the convention: egs3.<tag>.dataset
# e.g. "mini_an4/esp2_asr" -> "egs3.mini_an4.esp2_asr.dataset"
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
        >>> resolve_dataset_module_name("mini_an4/esp2_asr")
        'egs3.mini_an4.esp2_asr.dataset'
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
        >>> module = _load_local_dataset_module("egs3/mini_an4/esp2_asr")
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


def _is_missing(error: ModuleNotFoundError, name: str) -> bool:
    """Whether ``error`` reports ``name`` or one of its parent packages missing."""
    return error.name is not None and (
        name == error.name or name.startswith(error.name + ".")
    )


def _locate_data_src(data_src: str) -> Any:
    """Locate a dotted ``data_src`` with Hydra, keeping dependency errors intact.

    ``hydra.utils.get_object`` wraps every import failure in an ``ImportError``.
    Only failures to find ``data_src`` itself are reported as such; an error
    raised while importing a module on the path, such as a missing dependency,
    is re-raised as the original exception.
    """
    try:
        return get_object(data_src)
    except ImportError as error:
        cause = error.__cause__
        if isinstance(cause, ModuleNotFoundError) and _is_missing(cause, data_src):
            if "." in data_src and cause.name == data_src:
                parent, leaf = data_src.rsplit(".", 1)
                raise ImportError(
                    f"Cannot resolve data_src {data_src!r}: {parent!r} has no "
                    f"attribute or submodule {leaf!r}",
                    name=parent,
                ) from None
            raise ModuleNotFoundError(
                f"Cannot resolve data_src {data_src!r}: no module named "
                f"{cause.name!r}",
                name=cause.name,
            ) from None
        if cause is None or isinstance(cause, AttributeError):
            raise
        raise cause from None


def resolve_data_src(
    data_src: str | None = None, recipe_dir: str | Path | None = None
) -> tuple[ModuleType, Any | None]:
    """Resolve a dataset source to its module and, if named, its dataset class.

    A dotted ``data_src`` is located with ``hydra.utils.get_object``, so it
    may name a module or an attribute of one. An error raised while importing
    a module on the path, such as a missing dependency of the dataset module,
    propagates unchanged.

    Args:
        data_src: Dataset source reference. See :func:`load_dataset_module`
            for the supported forms.
        recipe_dir: Recipe root directory used for local module loading.

    Returns:
        ``(module, dataset_cls)``. When ``data_src`` names a module,
        ``dataset_cls`` is ``None`` and the module's ``Dataset`` attribute is
        the dataset class. Otherwise ``dataset_cls`` is the named attribute,
        not yet validated, and ``module`` is the module that holds it.

    Raises:
        AssertionError: If both ``data_src`` and ``recipe_dir`` are ``None``.
        ModuleNotFoundError: If a package or module on the path is missing.
        ImportError: If the last component of ``data_src`` is neither an
            attribute nor a submodule of its parent.

    Examples:
        >>> resolve_data_src("egs3.mini_an4.esp2_asr.dataset")[1] is None
        True
        >>> module, dataset_cls = resolve_data_src(
        ...     "egs3.libritts.f5tts.dataset.librispeech_pc.LibriSpeechPCDataset"
        ... )
        >>> module.__name__, dataset_cls.__name__
        ('egs3.libritts.f5tts.dataset.librispeech_pc', 'LibriSpeechPCDataset')
    """
    if data_src is None:
        assert recipe_dir is not None, "recipe_dir must be set when data_src is None."

        # Local recipe mode: load <recipe_dir>/dataset/__init__.py directly.
        return _load_local_dataset_module(recipe_dir), None
    if _is_tag(data_src):

        # Tag mode: "mini_an4/esp2_asr" -> "egs3.mini_an4.esp2_asr.dataset"
        return import_module(resolve_dataset_module_name(data_src)), None

    target = _locate_data_src(data_src)
    if isinstance(target, ModuleType):
        return target, None
    module = sys.modules.get(data_src.rsplit(".", 1)[0]) or inspect.getmodule(target)
    return module, target


def load_dataset_module(
    data_src: str | None = None, recipe_dir: str | Path | None = None
):
    """Load a dataset module from a dataset source or local recipe path.

    Args:
        data_src: Dataset source reference. Supported forms:
            - ``None``: load local ``recipe_dir/dataset`` module.
            - tag form: ``mini_an4/esp2_asr``.
            - module path form: ``egs3.mini_an4.esp2_asr.dataset``.
            - class path form:
              ``egs3.libritts.f5tts.dataset.librispeech_pc.LibriSpeechPCDataset``;
              the module that defines the class is returned.
        recipe_dir: Recipe root directory used for local module loading.

    Returns:
        Imported dataset module object.

    Raises:
        AssertionError: If both ``data_src`` and ``recipe_dir`` are ``None``.
        ModuleNotFoundError: If target module cannot be found.
        ImportError: If import fails for other reasons, or a class path names
            an attribute its module does not have.

    Notes:
        This function does not instantiate ``Dataset`` or ``DatasetBuilder``.
        It only resolves and imports the module.

    Examples:
        >>> module = load_dataset_module(data_src="mini_an4/esp2_asr")
        >>> hasattr(module, "DatasetBuilder")
        True
        >>> local = load_dataset_module(
        ...     data_src=None, recipe_dir="egs3/mini_an4/esp2_asr"
        ... )
        >>> hasattr(local, "Dataset")
        True
        >>> load_dataset_module(data_src=None, recipe_dir=None)
        Traceback (most recent call last):
        ...
        AssertionError: recipe_dir must be set when data_src is None.
    """
    return resolve_data_src(data_src=data_src, recipe_dir=recipe_dir)[0]


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
        ...     {"data_src": "mini_an4/esp2_asr", "data_src_args": {"split": "train"}}
        ... )
        ('mini_an4/esp2_asr', {'split': 'train'})
        >>> parse_dataset_reference_config({"data_src_args": {"split": "test"}})
        (None, {'split': 'test'})
    """
    plain = _to_plain_dict(config)
    data_src = plain.get(_DATA_SRC_KEY)
    if data_src is not None:
        if not isinstance(data_src, str):
            raise TypeError("`data_src` must be a string or None.")
        data_src = data_src.strip() or None
    data_src_args = plain.get(_DATA_SRC_ARGS_KEY, {})
    if data_src_args is None:
        data_src_args = {}
    data_src_args = dict(data_src_args)
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
        Instantiated dataset object: ``module.Dataset(**data_src_args)``, or
        ``DatasetClass(**data_src_args)`` when ``data_src`` is a class path.

    Raises:
        AttributeError: If the target module does not expose ``Dataset``.
        ModuleNotFoundError: If dataset module resolution fails.
        ImportError: If a class path names an attribute its module lacks.
        TypeError: If ``data_src_args`` is incompatible with dataset
            constructor, or the dataset class is not a class inheriting
            ``BaseDataset``.

    Notes:
        Only ``data_src_args`` is forwarded to the dataset constructor.
        Top-level config fields other than ``data_src_args`` are not passed
        through.

    Examples:
        >>> cfg = {"data_src": "mini_an4/esp2_asr", "data_src_args": {"split": "train"}}
        >>> ds = instantiate_dataset_reference(cfg, recipe_dir="egs3/mini_an4/esp2_asr")
        >>> hasattr(ds, "__len__")
        True

        Local dataset module loading (no data_src):
        >>> cfg = {"data_src_args": {"split": "test"}}
        >>> _ = instantiate_dataset_reference(cfg, recipe_dir="egs3/mini_an4/esp2_asr")

        Dataset class named directly:
        >>> cfg = {
        ...     "data_src": (
        ...         "egs3.libritts.f5tts.dataset.librispeech_pc.LibriSpeechPCDataset"
        ...     ),
        ...     "data_src_args": {"manifest_path": "data/librispeech_pc/manifest.tsv"},
        ... }
        >>> ds = instantiate_dataset_reference(cfg)
    """
    data_src, data_src_args = parse_dataset_reference_config(config)
    module, dataset_cls = resolve_data_src(data_src=data_src, recipe_dir=recipe_dir)
    if dataset_cls is None:
        dataset_cls = getattr(module, "Dataset")
    if not isinstance(dataset_cls, type):
        raise TypeError(
            f"data_src {data_src!r} names {dataset_cls!r}, which is not a class; "
            "it must name a dataset class inheriting BaseDataset"
        )
    if not issubclass(dataset_cls, BaseDataset):
        raise TypeError(
            f"{dataset_cls.__qualname__} does not inherit BaseDataset; a "
            "recipe's Dataset class must declare `fields` through it"
        )
    return dataset_cls(**data_src_args)
