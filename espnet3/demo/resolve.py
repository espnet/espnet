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

If a system does not follow these conventions, infer.yaml must set explicit
``provider._target_`` and/or ``runner._target_`` so the resolver can load the
correct implementation.
"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any, Dict

from hydra.utils import get_class, get_method
from omegaconf import DictConfig, OmegaConf

from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def load_demo_config(demo_dir: Path) -> DictConfig:
    """Load demo.yaml from a demo directory.

    Args:
        demo_dir: Directory containing a demo.yaml file.
    Returns:
        DictConfig produced by load_config_with_defaults.
    Raises:
        FileNotFoundError: When demo.yaml does not exist.

    Example:
        >>> from pathlib import Path
        >>> cfg = load_demo_config(Path("exp/demo"))
        >>> cfg.system  # doctest: +SKIP
        'asr'
    """
    return load_config_with_defaults(str(demo_dir / "demo.yaml"))


def resolve_infer_path(infer_config, demo_cfg_path: Path | None) -> Path | None:
    """Resolve infer_config path relative to the demo config directory.

    Args:
        infer_config: Value of demo_cfg.infer_config (string or Path-like).
        demo_cfg_path: Path to the demo.yaml file, if available.
    Returns:
        Absolute Path to the inference config, or None if infer_config is empty.

    Example:
        >>> from pathlib import Path
        >>> resolve_infer_path("infer.yaml", Path("exp/demo/demo.yaml"))
        Path('.../exp/demo/infer.yaml')  # doctest: +ELLIPSIS
    """
    if not infer_config:
        return None
    return resolve_absolute_path(
        infer_config,
        base=demo_cfg_path.parent if demo_cfg_path is not None else Path.cwd(),
    )


def load_infer_config(infer_path: Path) -> DictConfig:
    """Load an inference config file and resolve OmegaConf references.

    Args:
        infer_path: Absolute path to an inference YAML file.
    Returns:
        DictConfig with all OmegaConf interpolations resolved.

    Example:
        >>> cfg = load_infer_config(Path("exp/demo/config/infer.yaml"))
        >>> isinstance(cfg, DictConfig)
        True
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

    Example:
        >>> from pathlib import Path
        >>> resolve_absolute_path("infer.yaml", base=Path("exp/demo"))
        Path('.../exp/demo/infer.yaml')  # doctest: +ELLIPSIS
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

    Example:
        >>> from omegaconf import OmegaConf
        >>> demo_cfg = OmegaConf.create({"output_keys": {"text": "hyp"}})
        >>> resolve_output_keys(demo_cfg)
        {'text': 'hyp'}
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

    Example:
        >>> from omegaconf import OmegaConf
        >>> demo_cfg = OmegaConf.create({"extra_kwargs": {"beam_size": 10}})
        >>> resolve_extra_kwargs(demo_cfg)
        {'beam_size': 10}
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


def resolve_infer_kwargs(infer_cfg: DictConfig | None) -> Dict[str, Any]:
    """Resolve inference runner kwargs derived from infer_config.

    This extracts inference-required settings like input keys and the output
    function path so demos can reuse the same infer.yaml configuration.

    Args:
        infer_cfg: Loaded inference config, or None.
    Returns:
        Mapping of keyword arguments to pass into runner.forward.
    """
    if infer_cfg is None:
        return {}
    mapping: Dict[str, Any] = {}
    input_key = getattr(infer_cfg, "input_key", None)
    if input_key is not None:
        mapping["input_key"] = input_key
    output_fn = getattr(infer_cfg, "output_fn", None)
    if output_fn:
        mapping["output_fn_path"] = output_fn
    return mapping


def resolve_provider_class(demo_cfg, infer_cfg: DictConfig | None = None):
    """Resolve inference provider class from infer.yaml or convention.

    Resolution order:
      1) infer_cfg.provider._target_ (or provider_class) if present.
      2) Convention-based path using demo_cfg.system.

    Conventions assume:
      ``espnet3.systems.<system>.inference.InferenceProvider`` exists.

    Args:
        demo_cfg: Demo configuration object.
        infer_cfg: Inference config object (optional).
    Returns:
        Provider class object, or None if no system is defined.

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({"system": "asr"})
        >>> resolve_provider_class(cfg)  # doctest: +SKIP
    """
    if infer_cfg is not None:
        provider_cfg = getattr(infer_cfg, "provider", None)
        if provider_cfg is not None:
            path = getattr(provider_cfg, "_target_", None) or getattr(
                provider_cfg, "provider_class", None
            )
            if path:
                return get_class(str(path))
    system = str(getattr(demo_cfg, "system", "")).lower()
    if not system:
        return None
    try:
        return get_class(f"espnet3.systems.{system}.inference.InferenceProvider")
    except Exception:
        logger.warning(
            "Provider class for system '%s' not found; using base InferenceProvider.",
            system,
        )
        return get_class("espnet3.systems.base.inference_provider.InferenceProvider")


def resolve_runner_class(demo_cfg, infer_cfg: DictConfig | None = None):
    """Resolve inference runner class from infer.yaml or convention.

    Resolution order:
      1) infer_cfg.runner._target_ (or runner_class) if present.
      2) Convention-based path using demo_cfg.system.

    Conventions assume:
      ``espnet3.systems.<system>.inference.InferenceRunner`` exists.

    Args:
        demo_cfg: Demo configuration object.
        infer_cfg: Inference config object (optional).
    Returns:
        Runner class object, or None if no system is defined.

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({"system": "asr"})
        >>> resolve_runner_class(cfg)  # doctest: +SKIP
    """
    if infer_cfg is not None:
        runner_cfg = getattr(infer_cfg, "runner", None)
        if runner_cfg is not None:
            path = getattr(runner_cfg, "_target_", None) or getattr(
                runner_cfg, "runner_class", None
            )
            if path:
                return get_class(str(path))
    system = str(getattr(demo_cfg, "system", "")).lower()
    if not system:
        return None
    try:
        return get_class(f"espnet3.systems.{system}.inference.InferenceRunner")
    except Exception:
        logger.warning(
            "Runner class for system '%s' not found; using base InferenceRunner.",
            system,
        )
        return get_class("espnet3.systems.base.inference_runner.InferenceRunner")


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
