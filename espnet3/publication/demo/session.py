"""Demo configuration, UI components and wiring to a recipe-provided callable.

The recipe constructs its backend and supplies a callable that accepts named
inputs and returns a mapping of named outputs. UI specs come explicitly from
``demo.yaml``; no model declarations or model loading are involved here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from espnet3.publication.demo.assets import DEFAULT_UI_ASSETS, UIAssetRegistry
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_front_matter(text: str) -> str:
    """Remove YAML front matter from a markdown string."""
    return _FRONT_MATTER_RE.sub("", text, count=1).lstrip("\n")


class DemoSession:
    """Keep a demo's configuration, callable, UI specs and assets together.

    Args:
        demo_dir: The packed demo directory.
        demo_cfg: Its loaded ``demo.yaml`` with explicit ui.inputs/ui.outputs.
        model: A recipe-provided callable taking named inputs and returning a
            mapping of named outputs in Gradio-compatible formats.
        registry: The UI assets a spec's ``type`` resolves through.
    """

    def __init__(
        self,
        demo_dir: Path,
        demo_cfg: DictConfig,
        model: Callable[..., dict[str, Any]],
        registry: UIAssetRegistry,
    ) -> None:
        """Keep the loaded parts together and resolve the UI specs."""
        self.demo_dir = demo_dir
        self.demo_cfg = demo_cfg
        self.model = model
        self.registry = registry

        title = self.demo_cfg.ui.get("title")
        self.title = str(title) if title is not None else None

        description = self.demo_cfg.ui.get("description")
        if not description:
            self.description = None
        else:
            path = Path(str(description))
            if not path.is_absolute():
                path = self.demo_dir / path
            if path.is_file():
                self.description = _strip_front_matter(path.read_text(encoding="utf-8"))
            else:
                self.description = str(description)

        model_cfg = self.demo_cfg.get("model") or {}
        mapping = model_cfg.get("call_args", OmegaConf.create({}))
        if not isinstance(mapping, DictConfig):
            raise TypeError("demo config model.call_args must be a mapping.")
        self.call_args = OmegaConf.to_container(mapping, resolve=True)

        self.input_specs = self._specs("inputs")
        self.output_specs = self._specs("outputs")
        if not self.output_specs:
            raise ValueError("Set ui.outputs in demo.yaml to name the outputs to show.")

    def _specs(self, which: str) -> list[dict[str, Any]]:
        """Read explicit UI specs without inspecting the model."""
        return [
            OmegaConf.to_container(spec, resolve=True)
            for spec in self.demo_cfg.ui.get(which, [])
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
        """Return the function the Run button calls.

        It takes the input components' values positionally, in the order
        of ``input_keys`` (or ``input_specs``), calls the model with them
        by field name, and returns the outputs named by ``output_keys`` (or
        ``output_specs``) in order - one value, or a list for several -
        already in Gradio-compatible formats supplied by the recipe.

        Args:
            input_specs: Resolved UI input specs; their ``key`` fields are
                the model's input names.
            output_specs: Resolved UI output specs.
            input_keys: Explicit input names, overriding ``input_specs``.
            output_keys: Explicit output names, overriding ``output_specs``.

        Returns:
            A function suitable for ``gr.Button.click(fn=...)``.

        Examples:
            >>> run = session.create_inference_fn(
            ...     session.input_specs, session.output_specs)
            >>> run("utt.wav")
            'hello world'
            >>> run = session.create_inference_fn(
            ...     input_keys=["speech"], output_keys=["text"])
        """
        input_keys = (
            list(input_keys)
            if input_keys is not None
            else [spec["key"] for spec in (input_specs or [])]
        )
        output_keys = (
            list(output_keys)
            if output_keys is not None
            else [spec["key"] for spec in (output_specs or [])]
        )

        def run_inference(*values: Any) -> Any:
            logger.info(
                "Demo inference | inputs=%s outputs=%s", input_keys, output_keys
            )
            try:
                result = self.model(**dict(zip(input_keys, values)), **self.call_args)
                outputs = [result[key] for key in output_keys]
                return outputs[0] if len(outputs) == 1 else outputs
            except Exception:
                logger.exception("Demo inference failed")
                raise

        return run_inference


def load_demo_session(
    demo_dir: str | Path,
    demo_config_path: str | Path,
    *,
    model: Callable[..., dict[str, Any]],
    registry: UIAssetRegistry | None = None,
) -> DemoSession:
    """Load a packed demo into a runtime session.

    Args:
        demo_dir: Packed demo directory.
        demo_config_path: The packed ``demo.yaml``; a relative path is taken
            from ``demo_dir``.
        model: Callable constructed by the recipe; no loader is selected here.
        registry: Optional recipe UI assets, cloned for the session.

    Returns:
        The loaded session.

    Raises:
        FileNotFoundError: If the demo directory or the config is missing.

    Examples:
        >>> session = load_demo_session("exp/demo", "demo.yaml", model=decode)
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
    logger.info("Loading demo session | demo_dir=%s config=%s", demo_root, config_path)
    demo_cfg = load_config_with_defaults(str(config_path))
    assets = DEFAULT_UI_ASSETS if registry is None else registry
    return DemoSession(demo_root, demo_cfg, model, assets.clone())
