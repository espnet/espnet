"""Demo session loading and runtime inference helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

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
    input/output wiring.

    Args:
        demo_dir: Packed demo directory.
        demo_cfg: Loaded packed demo config.
        model: Loaded inference model.
        registry: Asset registry for this demo session.
    """

    def __init__(
        self,
        demo_dir: Path,
        demo_cfg: DictConfig,
        model: InferenceModel,
        registry: UIAssetRegistry,
    ) -> None:
        """Initialize from already-loaded demo config, model, and registry."""
        self.demo_dir = demo_dir
        self.demo_cfg = demo_cfg
        self.model = model
        self.registry = registry

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

        model_cfg = getattr(self.demo_cfg, "model", None)
        mapping = (
            model_cfg.get("call_args", OmegaConf.create({}))
            if model_cfg is not None
            else OmegaConf.create({})
        )
        if not isinstance(mapping, DictConfig):
            raise TypeError("demo config model.call_args must be a mapping.")
        call_args = OmegaConf.to_container(mapping, resolve=True)
        self.call_args = call_args

        self.input_specs = [
            OmegaConf.to_container(spec, resolve=True)
            for spec in self.demo_cfg.ui.inputs
        ]
        self.output_specs = [
            OmegaConf.to_container(spec, resolve=True)
            for spec in self.demo_cfg.ui.outputs
        ]

    def build_input_component(self, spec: dict[str, Any]) -> Any:
        """Build one Gradio input component from a spec."""
        return self.registry.get(spec["type"]).build_input(spec)

    def build_output_component(self, spec: dict[str, Any]) -> Any:
        """Build one Gradio output component from a spec."""
        return self.registry.get(spec["type"]).build_output(spec)

    def create_inference_fn(
        self,
        input_specs: list[dict[str, Any]] | None = None,
        output_specs: list[dict[str, Any]] | None = None,
        input_keys: list[str] | None = None,
        output_keys: list[str] | None = None,
    ):
        """Return a callable that maps Gradio values to model inference.

        Args:
            input_specs: Optional resolved UI input spec list.
            output_specs: Optional resolved UI output spec list.
            input_keys: Optional model input key list. When omitted, keys are
                derived from ``input_specs``.
            output_keys: Optional model output key list. When omitted, keys are
                derived from ``output_specs``.

        Returns:
            Callable: Function suitable for ``gr.Button.click`` that
            maps UI values to model input keys and runs inference.
        """
        resolved_input_specs = input_specs or []
        resolved_output_specs = output_specs or []
        resolved_input_keys = (
            input_keys
            if input_keys is not None
            else [spec["key"] for spec in resolved_input_specs]
        )
        resolved_output_keys = (
            output_keys
            if output_keys is not None
            else [spec["key"] for spec in resolved_output_specs]
        )

        def run_inference(*values: Any) -> Any:
            logger.info(
                "Demo inference start | num_inputs=%d input_keys=%s output_keys=%s",
                len(values),
                resolved_input_keys,
                resolved_output_keys,
            )
            try:
                item = {}
                for key, value in zip(resolved_input_keys, values):
                    item[key] = value
                logger.info(
                    "Calling inference model | input_keys=%s call_args=%s",
                    list(item.keys()),
                    self.call_args,
                )
                result = self.model(item, **self.call_args)
                logger.info(
                    "Inference model returned | output_keys=%s",
                    list(result.keys()) if isinstance(result, dict) else None,
                )
                outputs = []
                for key in resolved_output_keys:
                    value = result[key] if isinstance(result, dict) else result
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
    demo_config_path: str | Path,
) -> DemoSession:
    """Load a packed demo into a runtime session.

    Args:
        demo_dir: Packed demo directory.
        demo_config_path: Explicit packed demo config path. Relative paths are
            resolved from ``demo_dir``.
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
    config_path = Path(demo_config_path)
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
    model = _build_demo_model(demo_cfg, demo_root)
    registry = DEFAULT_UI_ASSETS.clone()
    logger.info(
        "Demo session ready | input_key=%s title=%s",
        getattr(model, "input_key", None),
        demo_cfg.ui.title,
    )
    return DemoSession(
        demo_root,
        demo_cfg,
        model,
        registry,
    )


def _build_demo_model(
    demo_cfg,
    demo_dir: Path,
) -> InferenceModel:
    model_cfg = demo_cfg.model
    dir_or_tag = model_cfg.get("dir_or_tag")
    if not dir_or_tag:
        raise ValueError("demo config must contain model.dir_or_tag.")
    model_trust_user_code = bool(model_cfg.get("trust_user_code", False))
    logger.info(
        "Building demo inference model | dir_or_tag=%s trust_user_code=%s "
        "base_dir=%s",
        dir_or_tag,
        model_trust_user_code,
        demo_dir,
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
        )
    return InferenceModel.from_pretrained(
        raw_ref,
        trust_user_code=model_trust_user_code,
    )
