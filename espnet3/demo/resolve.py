"""Demo configuration and resolution helpers.

This module centralizes demo config parsing and class/method resolution for
the demo runtime and pack pipeline. It intentionally relies on module path
conventions for ESPnet3 systems so callers can omit explicit imports in
demo.yaml when the system name is known.

Expected module path conventions (by system name):
  - Provider class:
      ``espnet3.systems.<system>.inference.InferenceProvider``
  - Runner class:
      ``espnet3.systems.<system>.inference.InferenceRunner``
  - Inference defaults function:
      ``espnet3.systems.<system>.demo.build_inference_default``

If a system does not follow these conventions, demo.yaml must set explicit
``inference.provider_class`` and/or ``inference.runner_class`` so the
resolver can load the correct implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from hydra.utils import get_class, get_method
from omegaconf import DictConfig, OmegaConf

from espnet3.utils.config import load_config_with_defaults


def load_demo_config(demo_dir: Path) -> DictConfig:
    """Load demo.yaml from a demo directory.

    Args:
        demo_dir: Directory containing a demo.yaml file.
    Returns:
        DictConfig produced by load_config_with_defaults.
    Raises:
        FileNotFoundError: When demo.yaml does not exist.
    """
    return load_config_with_defaults(str(demo_dir / "demo.yaml"))


def resolve_infer_path(infer_config) -> Path | None:
    """Resolve infer_config path relative to the current working directory.

    Args:
        infer_config: Value of demo_cfg.infer_config (string or Path-like).
    Returns:
        Absolute Path to the inference config, or None if infer_config is empty.
    """
    if not infer_config:
        return None
    return resolve_absolute_path(
        infer_config,
        base=Path.cwd(),
    )


def load_infer_config(infer_path: Path) -> DictConfig:
    """Load an inference config file and resolve OmegaConf references.

    Args:
        infer_path: Absolute path to an inference YAML file.
    Returns:
        DictConfig with all OmegaConf interpolations resolved.
    """
    return OmegaConf.create(
        OmegaConf.to_container(load_config_with_defaults(str(infer_path)), resolve=True)
    )


def resolve_absolute_path(path_value, *, base: Path) -> Path:
    """Return an absolute path for a string/Path value relative to base.

    Args:
        path_value: Path-like value to resolve.
        base: Base directory used when path_value is relative.
    Returns:
        Absolute Path.
    Raises:
        ValueError: If path_value is None.
    """
    if path_value is None:
        raise ValueError("absolute path could not be resolved.")
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def resolve_output_keys(demo_cfg) -> Dict[str, str]:
    """Resolve output key mapping from demo.yaml or system defaults.

    The output key map connects UI output component names to keys in the
    inference result dictionary (e.g., {"text": "hyp"}).

    Args:
        demo_cfg: Demo configuration object.
    Returns:
        Mapping of UI output names to result dict keys. Empty when unset.
    """
    mapping = getattr(demo_cfg, "output_keys", None)
    if mapping is not None:
        if isinstance(mapping, DictConfig):
            return OmegaConf.to_container(mapping, resolve=True) or {}
        return dict(mapping)
    defaults = _load_system_inference_defaults(demo_cfg)
    if defaults and defaults.get("output_keys"):
        return dict(defaults["output_keys"])
    return {}


def resolve_extra_kwargs(demo_cfg) -> Dict[str, Any]:
    """Resolve extra keyword arguments for the inference runner.

    Extra kwargs are passed directly to the inference runner and are not
    derived from UI inputs. Use this for constant settings such as
    beam size or decoding options.

    Args:
        demo_cfg: Demo configuration object.
    Returns:
        Mapping of keyword arguments to pass into runner.forward.
    """
    mapping = getattr(demo_cfg, "extra_kwargs", None)
    if mapping is not None:
        if isinstance(mapping, DictConfig):
            return OmegaConf.to_container(mapping, resolve=True) or {}
        return dict(mapping)
    defaults = _load_system_inference_defaults(demo_cfg)
    if defaults and defaults.get("extra_kwargs"):
        return dict(defaults["extra_kwargs"])
    return {}


def resolve_provider_class(demo_cfg):
    """Resolve inference provider class from demo.yaml or system convention.

    Resolution order:
      1) demo_cfg.inference.provider_class if present.
      2) Convention-based path using demo_cfg.system.

    Conventions assume:
      ``espnet3.systems.<system>.inference.InferenceProvider`` exists.

    Args:
        demo_cfg: Demo configuration object.
    Returns:
        Provider class object, or None if no system is defined.
    """
    path = getattr(getattr(demo_cfg, "inference", None), "provider_class", None)
    if path:
        return get_class(str(path))
    system = str(getattr(demo_cfg, "system", "")).lower()
    if not system:
        return None
    return get_class(f"espnet3.systems.{system}.inference.InferenceProvider")


def resolve_runner_class(demo_cfg):
    """Resolve inference runner class from demo.yaml or system convention.

    Resolution order:
      1) demo_cfg.inference.runner_class if present.
      2) Convention-based path using demo_cfg.system.

    Conventions assume:
      ``espnet3.systems.<system>.inference.InferenceRunner`` exists.

    Args:
        demo_cfg: Demo configuration object.
    Returns:
        Runner class object, or None if no system is defined.
    """
    path = getattr(getattr(demo_cfg, "inference", None), "runner_class", None)
    if path:
        return get_class(str(path))
    system = str(getattr(demo_cfg, "system", "")).lower()
    if not system:
        return None
    return get_class(f"espnet3.systems.{system}.inference.InferenceRunner")


def _load_system_inference_defaults(demo_cfg):
    """Resolve system-level inference defaults via convention-based import.

    This expects each system to expose a function at:
      ``espnet3.systems.<system>.demo.build_inference_default``
    and returns the dict it produces.

    Args:
        demo_cfg: Demo configuration object.
    Returns:
        Dict of default inference settings, or None if not available.
    """
    system = str(getattr(demo_cfg, "system", "")).lower()
    if not system:
        return None
    try:
        fn = get_method(f"espnet3.systems.{system}.demo.build_inference_default")
    except Exception:
        return None
    return fn()
