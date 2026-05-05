"""Demo session loading and runtime inference helpers."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

import espnet3.publication.demo.assets as _assets_mod
from espnet3.publication.demo.assets import (
    DEFAULT_UI_ASSETS,
    UIAssetRegistry,
)
from espnet3.publication.inference_model import InferenceModel
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


class DemoSession:
    """Loaded demo runtime state for a packed demo directory.

    This class keeps the packed demo config, resolved model, and asset
    registry together so recipe-local ``demo.py`` files can build any Gradio
    layout they want while still reusing ESPnet's model loading and
    input/output normalization.

    Args:
        demo_dir: Packed demo directory.
        demo_cfg: Loaded packed demo config.
        model: Loaded inference model.
        registry: Asset registry for this demo session.
        model_overrides: Optional runtime overrides already applied during
            model construction.
    """

    def __init__(
        self,
        demo_dir: Path,
        demo_cfg: DictConfig,
        model: InferenceModel,
        registry: UIAssetRegistry,
        model_overrides: dict[str, Any] | None = None,
    ) -> None:
        """Initialize from already-loaded demo config, model, and registry."""
        self.demo_dir = demo_dir
        self.demo_cfg = demo_cfg
        self.model = model
        self.registry = registry
        self.model_overrides = model_overrides or {}

        title = self.demo_cfg.ui.title
        self.title = str(title) if title is not None else None

        description = self.demo_cfg.ui.description
        if not description:
            self.description = None
        else:
            path = Path(str(description))
            if not path.is_absolute():
                path = self.demo_dir / path
            self.description = (
                path.read_text(encoding="utf-8") if path.is_file() else str(description)
            )

        mapping = self.demo_cfg.inference_args
        if not isinstance(mapping, DictConfig):
            raise TypeError("demo config inference_args must be a mapping.")
        inference_args = OmegaConf.to_container(mapping, resolve=True)
        if not isinstance(inference_args, dict):
            raise TypeError("demo config inference_args must resolve to a dict.")
        self.inference_args = inference_args

        self.input_specs = [
            _resolve_spec_dict(spec, "ui.inputs") for spec in self.demo_cfg.ui.inputs
        ]
        self.output_specs = [
            _resolve_spec_dict(spec, "ui.outputs") for spec in self.demo_cfg.ui.outputs
        ]

    def build_input_component(self, spec: dict[str, Any]) -> Any:
        """Build one Gradio input component from a spec."""
        return self.registry.get(spec["type"]).build_input(spec)

    def build_output_component(self, spec: dict[str, Any]) -> Any:
        """Build one Gradio output component from a spec."""
        return self.registry.get(spec["type"]).build_output(spec)

    def create_inference_fn(
        self,
        input_specs: list[dict[str, Any]],
        output_specs: list[dict[str, Any]],
    ):
        """Return a callable that maps Gradio values to model inference.

        Args:
            input_specs: Resolved UI input spec list.
            output_specs: Resolved UI output spec list.

        Returns:
            Callable: Function suitable for ``gr.Button.click`` that
            normalizes UI inputs, runs inference, and formats outputs.
        """

        def run_inference(*values: Any) -> Any:
            logger.info(
                "Demo inference start | num_inputs=%d input_keys=%s output_keys=%s",
                len(values),
                [spec["key"] for spec in input_specs],
                [spec["key"] for spec in output_specs],
            )
            try:
                item = {}
                for spec, value in zip(input_specs, values):
                    key = spec["key"]
                    item[key] = self.registry.get(spec["type"]).normalize_input(
                        value, spec
                    )
                logger.info(
                    "Calling inference model | inputs=%s inference_args=%s",
                    {k: _summarize_demo_value(v) for k, v in item.items()},
                    self.inference_args,
                )
                result = self.model(item, **self.inference_args)
                logger.info(
                    "Inference model returned | result=%s",
                    _summarize_demo_value(result),
                )
                outputs = []
                for spec in output_specs:
                    key = spec["key"]
                    value = result[key] if isinstance(result, dict) else result
                    value = self.registry.get(spec["type"]).format_output(value, spec)
                    outputs.append(value)
                if len(outputs) == 1:
                    outputs = outputs[0]
                return outputs
            except Exception:
                logger.exception("Demo inference failed")
                raise

        return run_inference


def load_demo_session(
    demo_dir: str | Path,
    demo_config_path: str | Path | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> DemoSession:
    """Load a packed demo into a runtime session.

    Args:
        demo_dir: Packed demo directory.
        demo_config_path: Optional explicit packed demo config path. When
            omitted, ``demo_dir / "demo.yaml"`` is used.
        model_overrides: Optional runtime overrides for the underlying
            inference config, such as ``{"device": "cuda"}``.

    Returns:
        DemoSession: Loaded runtime session for the packed demo.

    Raises:
        FileNotFoundError: If the demo directory or packed demo config is
            missing.
    """
    demo_root = Path(demo_dir).resolve()
    if not demo_root.is_dir():
        raise FileNotFoundError(
            f"demo_dir must point to an existing directory: {demo_root}"
        )
    config_path = (
        Path(demo_config_path)
        if demo_config_path is not None
        else demo_root / "demo.yaml"
    )
    if not config_path.is_absolute():
        config_path = demo_root / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"demo config path does not exist: {config_path}")
    logger.info(
        "Loading demo session | demo_dir=%s demo_config_path=%s",
        demo_root,
        config_path,
    )
    demo_cfg = load_config_with_defaults(str(config_path))
    logger.info("Loaded demo config | path=%s", config_path)
    model = _build_demo_model(demo_cfg, demo_root, model_overrides=model_overrides)
    registry = DEFAULT_UI_ASSETS.clone()
    _load_recipe_ui_assets(registry, demo_root, demo_cfg)
    logger.info(
        "Demo session ready | input_key=%s resolved_device=%s title=%s",
        getattr(model, "input_key", None),
        getattr(model, "resolved_device", None),
        demo_cfg.ui.title,
    )
    return DemoSession(
        demo_root,
        demo_cfg,
        model,
        registry,
        model_overrides=model_overrides,
    )


def build_runtime_overrides(
    override_args: list[str] | None = None,
    base_overrides: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build one runtime override mapping for the packed inference config.

    Args:
        override_args: Optional dotlist overrides such as
            ``["model.beam_size=1"]``.
        base_overrides: Optional base mapping merged before ``override_args``.

    Returns:
        dict[str, Any] | None: Resolved override mapping, or None when no
        override entries were provided.

    Examples:
        >>> build_runtime_overrides(
        ...     override_args=["model.beam_size=1"],
        ...     base_overrides={"device": "cpu"},
        ... )
        {'model': {'beam_size': 1}, 'device': 'cpu'}
    """
    override_cfg = OmegaConf.create(base_overrides or {})
    if override_args:
        override_cfg = OmegaConf.merge(
            override_cfg, OmegaConf.from_dotlist(override_args)
        )
    resolved = OmegaConf.to_container(override_cfg, resolve=True) or {}
    return resolved if resolved else None


def _build_demo_model(
    demo_cfg,
    demo_dir: Path,
    model_overrides: dict[str, Any] | None = None,
) -> InferenceModel:
    model_cfg = demo_cfg.model
    dir_or_tag = model_cfg.dir_or_tag
    if not dir_or_tag:
        raise ValueError("demo config must contain model.dir_or_tag.")
    model_trust_user_code = bool(model_cfg.trust_user_code)
    logger.info(
        "Building demo inference model | dir_or_tag=%s trust_user_code=%s "
        "base_dir=%s overrides=%s",
        dir_or_tag,
        model_trust_user_code,
        demo_dir,
        model_overrides,
    )
    raw_ref = str(dir_or_tag)
    candidate = Path(raw_ref).expanduser()
    resolved_candidate = (
        (demo_dir / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    if resolved_candidate.exists():
        if not resolved_candidate.is_dir():
            raise FileNotFoundError(
                "model.dir_or_tag resolved to a filesystem path, but it is not "
                f"a directory: {resolved_candidate}"
            )
        return InferenceModel.from_packed(
            resolved_candidate,
            trust_user_code=model_trust_user_code,
            config_overrides=model_overrides,
        )
    return InferenceModel.from_pretrained(
        raw_ref,
        trust_user_code=model_trust_user_code,
        config_overrides=model_overrides,
    )


def _load_recipe_ui_assets(
    registry: UIAssetRegistry,
    demo_dir: Path,
    demo_cfg,
) -> None:
    registry_path = demo_cfg.ui.asset_registry
    if not registry_path:
        logger.info("No recipe UI asset registry configured")
        return
    path = Path(str(registry_path))
    ui_module_path = demo_dir / (Path(path.name) if path.is_absolute() else path)
    if not ui_module_path.is_file():
        logger.warning(
            "Recipe UI asset registry not found in bundle: %s", ui_module_path
        )
        return
    logger.info("Loading recipe UI asset registry: %s", ui_module_path)

    previous_registry = _assets_mod._ACTIVE_REGISTRY
    _assets_mod._ACTIVE_REGISTRY = registry

    module_name = f"_espnet3_demo_ui_{abs(hash(ui_module_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, ui_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load demo UI module: {ui_module_path}")
    module = importlib.util.module_from_spec(spec)
    demo_dir_str = str(demo_dir)
    should_pop_path = demo_dir_str not in sys.path
    if should_pop_path:
        sys.path.insert(0, demo_dir_str)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        register_fn = getattr(module, "register_assets", None)
        if register_fn is not None:
            register_fn(registry)
            logger.info("Registered recipe UI assets via register_assets()")
    finally:
        _assets_mod._ACTIVE_REGISTRY = previous_registry
        sys.modules.pop(module_name, None)
        if should_pop_path:
            try:
                sys.path.remove(demo_dir_str)
            except ValueError:
                pass


def _resolve_spec_dict(spec: DictConfig, field_name: str) -> dict[str, Any]:
    resolved = OmegaConf.to_container(spec, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"demo config {field_name} entries must resolve to dicts.")
    return resolved


def _summarize_demo_value(value: Any) -> str:
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        np = None

    if np is not None and isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())})"
    if isinstance(value, (list, tuple)):
        preview = ", ".join(type(v).__name__ for v in list(value)[:3])
        return f"{type(value).__name__}(len={len(value)}, items=[{preview}])"
    if isinstance(value, str):
        return f"str(len={len(value)})"
    return type(value).__name__
