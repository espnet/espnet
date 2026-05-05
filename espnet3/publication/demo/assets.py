"""Demo UI asset registry and runtime helpers."""

from __future__ import annotations

import importlib.util
import logging
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from espnet3.publication.demo.config import load_demo_config
from espnet3.publication.inference_model import InferenceModel

logger = logging.getLogger(__name__)


class UIAsset:
    """Base class for one demo UI asset type."""

    def build_input(self, spec: dict[str, Any]) -> Any:
        raise ValueError(
            f"{self.__class__.__name__} does not support input components."
        )

    def build_output(self, spec: dict[str, Any]) -> Any:
        raise ValueError(
            f"{self.__class__.__name__} does not support output components."
        )

    def normalize_input(self, value: Any, spec: dict[str, Any]) -> Any:
        _ = spec
        return value

    def format_output(self, value: Any, spec: dict[str, Any]) -> Any:
        _ = spec
        return value


AssetRegistration = UIAsset | type[UIAsset]


class DefaultAudioUI(UIAsset):
    """Default Gradio audio asset."""

    def build_input(self, spec: dict[str, Any]) -> Any:
        import gradio as gr

        component_args = _resolve_component_args(spec)
        component_args.setdefault("type", "numpy")
        return gr.Audio(**component_args)

    def build_output(self, spec: dict[str, Any]) -> Any:
        import gradio as gr

        return gr.Audio(**_resolve_component_args(spec))

    def normalize_input(self, value: Any, spec: dict[str, Any]) -> Any:
        _ = spec
        import numpy as np

        if isinstance(value, (list, tuple)) and len(value) == 2:
            _, audio = value
            if isinstance(audio, np.ndarray):
                return audio.astype(np.float32)
        return value


class DefaultTextUI(UIAsset):
    """Default Gradio text asset."""

    def build_input(self, spec: dict[str, Any]) -> Any:
        import gradio as gr

        return gr.Textbox(**_resolve_component_args(spec))

    def build_output(self, spec: dict[str, Any]) -> Any:
        import gradio as gr

        return gr.Textbox(**_resolve_component_args(spec))


class UIAssetRegistry:
    """Registry of named UI asset definitions."""

    def __init__(self, assets: dict[str, UIAsset] | None = None) -> None:
        self._assets = dict(assets or {})

    def register(
        self,
        name: str,
        asset: AssetRegistration,
        replace: bool = False,
    ) -> None:
        """Register one asset definition."""
        if not replace and name in self._assets:
            raise ValueError(f"UI asset already registered: {name}")
        self._assets[name] = _coerce_asset(asset)

    def get(self, name: str) -> UIAsset:
        """Return one registered asset."""
        if name not in self._assets:
            raise KeyError(f"Unknown UI asset type: {name}")
        return self._assets[name]

    def clone(self) -> "UIAssetRegistry":
        """Return a shallow copy for one demo session."""
        return UIAssetRegistry(self._assets)


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
    """

    def __init__(
        self,
        demo_dir: Path,
        demo_cfg: DictConfig,
        model: InferenceModel,
        registry: UIAssetRegistry,
        model_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.demo_dir = demo_dir
        self.demo_cfg = demo_cfg
        self.model = model
        self.registry = registry
        self.model_overrides = model_overrides or {}

    @property
    def ui_cfg(self):
        return getattr(self.demo_cfg, "ui", None)

    @property
    def title(self) -> str | None:
        """Return the configured demo title."""
        if self.ui_cfg is None:
            return None
        title = getattr(self.ui_cfg, "title", None)
        return str(title) if title is not None else None

    @property
    def description(self) -> str | None:
        """Return inline markdown or markdown file content for the demo."""
        return _resolve_description(self.ui_cfg, self.demo_dir)

    @property
    def inference_args(self) -> dict[str, Any]:
        """Return fixed inference kwargs from the demo config."""
        return _resolve_inference_args(self.demo_cfg)

    def resolve_input_specs(self) -> list[dict[str, Any]]:
        """Return configured input specs or the default single-audio input."""
        return _resolve_input_specs(self.ui_cfg, self.model)

    def resolve_output_specs(self) -> list[dict[str, Any]]:
        """Return configured output specs or the default text output."""
        return _resolve_output_specs(self.ui_cfg)

    def build_input_component(self, spec: dict[str, Any]) -> Any:
        """Build one Gradio input component from a spec."""
        return self.registry.get(spec.get("type", "audio")).build_input(spec)

    def build_output_component(self, spec: dict[str, Any]) -> Any:
        """Build one Gradio output component from a spec."""
        return self.registry.get(spec.get("type", "text")).build_output(spec)

    def create_inference_fn(
        self,
        input_specs: list[dict[str, Any]],
        output_specs: list[dict[str, Any]],
    ):
        """Return a callable that maps Gradio values to model inference."""

        def run_inference(*values: Any) -> Any:
            _emit_demo_message(
                "info",
                "Demo inference start | num_inputs=%d input_keys=%s output_keys=%s",
                len(values),
                [spec.get("key") for spec in input_specs],
                [spec.get("key") for spec in output_specs],
            )
            try:
                item = {}
                for spec, value in zip(input_specs, values):
                    key = spec["key"]
                    asset = self.registry.get(spec.get("type", "audio"))
                    _emit_demo_message(
                        "info",
                        "Normalizing demo input | key=%s type=%s raw=%s",
                        key,
                        spec.get("type", "audio"),
                        _summarize_value(value),
                    )
                    item[key] = asset.normalize_input(value, spec)
                    _emit_demo_message(
                        "info",
                        "Normalized demo input | key=%s value=%s",
                        key,
                        _summarize_value(item[key]),
                    )

                _emit_demo_message(
                    "info",
                    "Calling inference model | inputs=%s inference_args=%s",
                    {key: _summarize_value(value) for key, value in item.items()},
                    self.inference_args,
                )
                result = self.model(item, **self.inference_args)
                _emit_demo_message(
                    "info",
                    "Inference model returned | result=%s",
                    _summarize_value(result),
                )
                outputs = _extract_outputs(self.registry, result, output_specs)
                _emit_demo_message(
                    "info",
                    "Demo outputs extracted | outputs=%s",
                    _summarize_value(outputs),
                )
                return outputs
            except Exception:
                logger.exception("Demo inference failed")
                print("Demo inference failed", flush=True)
                traceback.print_exc()
                raise

        return run_inference


_ACTIVE_REGISTRY: UIAssetRegistry | None = None
DEFAULT_UI_ASSETS = UIAssetRegistry()


def register_asset(
    name: str,
    asset: AssetRegistration,
    replace: bool = False,
) -> None:
    """Register a UI asset in the active recipe registry or the default one."""
    registry = _ACTIVE_REGISTRY or DEFAULT_UI_ASSETS
    registry.register(name, asset, replace=replace)


def load_demo_session(
    demo_dir: str | Path,
    demo_config_path: str | Path | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> DemoSession:
    """Load a packed demo into a runtime session.

    Args:
        demo_dir: Packed demo directory.
        demo_config_path: Optional explicit packed demo config path.
        model_overrides: Optional runtime overrides for the underlying
            inference config, such as ``{"device": "cuda"}``.

    Returns:
        Loaded demo session.
    """
    demo_root = Path(demo_dir).resolve()
    _emit_demo_message(
        "info",
        "Loading demo session | demo_dir=%s demo_config_path=%s",
        demo_root,
        demo_config_path,
    )
    config_path, demo_cfg = load_demo_config(demo_root, demo_config_path)
    _emit_demo_message("info", "Loaded demo config | path=%s", config_path)
    model = _build_demo_model(
        demo_cfg,
        demo_root,
        model_overrides=model_overrides,
    )
    registry = DEFAULT_UI_ASSETS.clone()
    _load_recipe_ui_assets(registry, demo_root, demo_cfg)
    _emit_demo_message(
        "info",
        "Demo session ready | input_key=%s resolved_device=%s title=%s",
        getattr(model, "input_key", None),
        getattr(model, "resolved_device", None),
        getattr(getattr(demo_cfg, "ui", None), "title", None),
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
    device: str | None = None,
    base_overrides: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build one runtime override mapping for the packed inference config."""
    override_cfg = OmegaConf.create(base_overrides or {})
    if override_args:
        override_cfg = OmegaConf.merge(
            override_cfg, OmegaConf.from_dotlist(override_args)
        )
    if device:
        override_cfg = OmegaConf.merge(
            override_cfg,
            OmegaConf.create({"device": device}),
        )
    resolved = OmegaConf.to_container(override_cfg, resolve=True) or {}
    return resolved if resolved else None


def setup_demo_assets(demo_dir: Path, demo_config) -> None:
    """Copy the launcher and optional recipe UI module into ``demo_dir``."""
    launcher_path = demo_dir / _resolve_launcher_name(demo_config)
    if not launcher_path.exists():
        shutil.copy2(_resolve_app_script(demo_config), launcher_path)
    _copy_recipe_ui_module(demo_dir, demo_config)


def _coerce_asset(asset: AssetRegistration) -> UIAsset:
    if isinstance(asset, UIAsset):
        return asset
    if isinstance(asset, type) and issubclass(asset, UIAsset):
        return asset()
    raise TypeError(
        "asset must be a UIAsset instance or UIAsset subclass, "
        f"but got: {type(asset)}"
    )


def _build_demo_model(
    demo_cfg,
    demo_dir: Path,
    model_overrides: dict[str, Any] | None = None,
) -> InferenceModel:
    model_cfg = getattr(demo_cfg, "model", None)
    dir_or_tag = getattr(model_cfg, "dir_or_tag", None) if model_cfg else None
    if not dir_or_tag:
        raise ValueError("demo config must contain model.dir_or_tag.")
    model_trust_user_code = bool(
        getattr(model_cfg, "trust_user_code", False) if model_cfg else False
    )
    _warn_if_unavailable_cuda(model_overrides)
    _emit_demo_message(
        "info",
        "Building demo inference model | dir_or_tag=%s trust_user_code=%s base_dir=%s overrides=%s",
        dir_or_tag,
        model_trust_user_code,
        demo_dir,
        model_overrides,
    )
    model = InferenceModel.from_dir_or_tag(
        dir_or_tag,
        trust_user_code=model_trust_user_code,
        base_dir=demo_dir,
        config_overrides=model_overrides,
    )
    _emit_backend_model_summary(model)
    return model


def _resolve_app_script(demo_config) -> Path:
    ui_cfg = getattr(demo_config, "ui", None)
    explicit = getattr(ui_cfg, "app_script", None) if ui_cfg else None
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"demo_config.ui.app_script not found: {path}")
        return path
    return Path(__file__).resolve().with_name("demo.py")


def _resolve_launcher_name(demo_config) -> str:
    pack_cfg = getattr(demo_config, "pack", None)
    launcher_name = getattr(pack_cfg, "launcher_name", None) if pack_cfg else None
    if launcher_name:
        return str(launcher_name)
    return _resolve_app_script(demo_config).name


def _resolve_bundle_path(path_value) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return Path(path.name)
    return path


def _copy_recipe_ui_module(demo_dir: Path, demo_config) -> None:
    ui_cfg = getattr(demo_config, "ui", None)
    registry_path = getattr(ui_cfg, "asset_registry", None) if ui_cfg else None
    if not registry_path:
        return
    src_path = Path(registry_path)
    if not src_path.is_absolute():
        src_path = (Path.cwd() / src_path).resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"demo_config.ui.asset_registry not found: {src_path}")
    dst_path = demo_dir / _resolve_bundle_path(registry_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    init_py = src_path.parent / "__init__.py"
    if init_py.is_file():
        shutil.copy2(init_py, dst_path.parent / "__init__.py")


def _load_recipe_ui_assets(
    registry: UIAssetRegistry,
    demo_dir: Path,
    demo_cfg,
) -> None:
    ui_cfg = getattr(demo_cfg, "ui", None)
    registry_path = getattr(ui_cfg, "asset_registry", None) if ui_cfg else None
    if not registry_path:
        _emit_demo_message("info", "No recipe UI asset registry configured")
        return
    ui_module_path = demo_dir / _resolve_bundle_path(registry_path)
    if not ui_module_path.is_file():
        _emit_demo_message(
            "warning",
            "Recipe UI asset registry not found in bundle: %s",
            ui_module_path,
        )
        return
    _emit_demo_message("info", "Loading recipe UI asset registry: %s", ui_module_path)

    global _ACTIVE_REGISTRY
    previous_registry = _ACTIVE_REGISTRY
    _ACTIVE_REGISTRY = registry

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
            _emit_demo_message(
                "info", "Registered recipe UI assets via register_assets()"
            )
    finally:
        _ACTIVE_REGISTRY = previous_registry
        sys.modules.pop(module_name, None)
        if should_pop_path:
            try:
                sys.path.remove(demo_dir_str)
            except ValueError:
                pass


def _resolve_input_specs(ui_cfg: Any, model: InferenceModel) -> list[dict[str, Any]]:
    inputs_cfg = getattr(ui_cfg, "inputs", None) if ui_cfg else None
    if inputs_cfg:
        return [OmegaConf.to_container(s, resolve=True) for s in inputs_cfg]
    primary_key = (
        model.input_key[0] if isinstance(model.input_key, list) else model.input_key
    )
    return [{"key": primary_key, "type": "audio", "label": "Input Audio"}]


def _resolve_output_specs(ui_cfg: Any) -> list[dict[str, Any]]:
    outputs_cfg = getattr(ui_cfg, "outputs", None) if ui_cfg else None
    if outputs_cfg:
        return [OmegaConf.to_container(s, resolve=True) for s in outputs_cfg]
    return [{"key": "hyp", "type": "text", "label": "Transcription"}]


def _resolve_inference_args(demo_cfg) -> dict[str, Any]:
    mapping = getattr(demo_cfg, "inference_args", None)
    if mapping is None:
        return {}
    if isinstance(mapping, DictConfig):
        return OmegaConf.to_container(mapping, resolve=True) or {}
    return dict(mapping)


def _extract_outputs(
    registry: UIAssetRegistry,
    result: Any,
    output_specs: list[dict[str, Any]],
) -> Any:
    outputs = []
    for spec in output_specs:
        key = spec.get("key")
        if isinstance(result, dict) and key in result:
            value = result.get(key)
        elif isinstance(result, dict):
            value = next(iter(result.values()), result)
        else:
            value = result
        value = registry.get(spec.get("type", "text")).format_output(value, spec)
        outputs.append(value)
    return outputs[0] if len(outputs) == 1 else outputs


def _resolve_description(ui_cfg: Any, demo_dir: Path) -> str | None:
    description = getattr(ui_cfg, "description", None) if ui_cfg else None
    if not description:
        return None
    path = Path(str(description))
    if not path.is_absolute():
        path = demo_dir / path
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return str(description)


def _resolve_component_args(spec: dict[str, Any]) -> dict[str, Any]:
    args = dict(spec.get("args", {}))
    if "label" not in args and "label" in spec:
        args["label"] = spec["label"]
    return args


def _summarize_value(value: Any) -> str:
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


def _emit_demo_message(level: str, message: str, *args: Any) -> None:
    text = message % args if args else message
    print(text, flush=True)
    log_fn = getattr(logger, level, logger.info)
    log_fn(text)


def _warn_if_unavailable_cuda(model_overrides: dict[str, Any] | None) -> None:
    if not model_overrides:
        return
    device = model_overrides.get("device")
    if not isinstance(device, str):
        return
    if not device.startswith("cuda"):
        return
    try:
        import torch
    except Exception:
        _emit_demo_message(
            "warning",
            "Requested device=%s but torch could not be imported. Inference may fail.",
            device,
        )
        return
    if torch.cuda.is_available():
        return
    _emit_demo_message(
        "warning",
        "Requested device=%s but CUDA is not available. Inference may fall back or run slowly.",
        device,
    )


def _emit_backend_model_summary(model: InferenceModel) -> None:
    """Print backend model attributes that help debug runtime overrides."""
    backend_model = getattr(model, "model", None)
    if backend_model is None:
        return

    beam_search = getattr(backend_model, "beam_search", None)
    beam_size = getattr(beam_search, "beam_size", None)
    maxlenratio = getattr(backend_model, "maxlenratio", None)
    minlenratio = getattr(backend_model, "minlenratio", None)
    runtime_device = getattr(backend_model, "device", None)
    _emit_demo_message(
        "info",
        (
            "Backend model summary | class=%s resolved_device=%s "
            "runtime_device=%s beam_size=%s maxlenratio=%s minlenratio=%s"
        ),
        type(backend_model).__name__,
        getattr(model, "resolved_device", None),
        runtime_device,
        beam_size,
        maxlenratio,
        minlenratio,
    )


register_asset("audio", DefaultAudioUI)
register_asset("text", DefaultTextUI)
