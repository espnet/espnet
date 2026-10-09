"""Demo session: a packed demo's config, its model, and the wiring between them.

A packed demo directory holds ``demo.yaml`` and the recipe's ``app.py``.
:func:`load_demo_session` reads the config, loads the model it names with
:func:`espnet3.api.inference.load`, and returns a :class:`DemoSession`
that the app builds its Gradio layout from: one input component per
declared input, one output component per declared output, and
:meth:`DemoSession.create_inference_fn` to bind the Run button to the
model. The specs default to the model's own declaration, so a
``demo.yaml`` need not repeat it::

    model:
      dir_or_tag: model_pack
    ui:
      title: My ASR demo
      description: README.md

Set ``ui.inputs`` / ``ui.outputs`` only to relabel a field or show a
subset.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from espnet3.api.inference import Audio, InferenceAPI, load
from espnet3.publication.demo.assets import DEFAULT_UI_ASSETS, UIAssetRegistry
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_front_matter(text: str) -> str:
    """Remove YAML front matter from a markdown string."""
    return _FRONT_MATTER_RE.sub("", text, count=1).lstrip("\n")


def to_ui(value: Any) -> Any:
    """Turn a contract value into what a Gradio output component takes.

    An :class:`Audio` becomes the ``(rate, samples)`` pair ``gr.Audio``
    shows, samples laid out ``(samples, channels)`` when there are several
    channels; anything else passes through.
    """
    if isinstance(value, Audio):
        array = value.array.T if value.array.ndim == 2 else value.array
        return value.rate, array
    return value


class DemoSession:
    """A packed demo, loaded: its config, its model and the UI specs.

    Recipe-local ``app.py`` files build any Gradio layout they want from
    this, reusing the model loading and the input/output wiring.

    Args:
        demo_dir: The packed demo directory.
        demo_cfg: Its loaded ``demo.yaml``.
        model: The loaded :class:`InferenceAPI`.
        registry: The UI assets a spec's ``type`` resolves through.

    Raises:
        TypeError: If ``demo.yaml`` sets ``model.call_args``: an
            ``Inference`` takes no call-time arguments - what it needs is in
            the bundle's ``inference.yaml``.

    Examples:
        >>> session = load_demo_session("demo", "demo/demo.yaml")
        >>> session.input_specs
        [{'key': 'speech', 'type': 'audio', 'label': 'Speech'}]
        >>> run = session.create_inference_fn(session.input_specs, session.output_specs)
        >>> run((16000, samples))
        'hello world'
    """

    def __init__(
        self,
        demo_dir: Path,
        demo_cfg: DictConfig,
        model: InferenceAPI,
        registry: UIAssetRegistry,
    ) -> None:
        """Keep the loaded parts together and resolve the UI specs."""
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
            if path.is_file():
                self.description = _strip_front_matter(path.read_text(encoding="utf-8"))
            else:
                self.description = str(description)

        model_cfg = self.demo_cfg.get("model") or {}
        if model_cfg.get("call_args"):
            raise TypeError(
                "demo.yaml sets model.call_args, but an Inference takes no "
                "call-time arguments; put them in the bundle's conf/inference.yaml"
            )

        self.input_specs = self._specs("inputs", model.inputs)
        self.output_specs = self._specs("outputs", model.outputs)
        if not self.output_specs:
            kinds = sorted({f.kind for f in model.outputs})
            raise ValueError(
                f"{type(model).__name__} declares no output the demo can show: "
                f"no UI asset is registered for {kinds}. Set ui.outputs in "
                "demo.yaml, or register an asset for the kind."
            )

    def _specs(self, which: str, fields) -> list[dict[str, Any]]:
        """``ui.<which>`` from the config, or one spec per declared field."""
        configured = self.demo_cfg.ui.get(which)
        if configured:
            return [OmegaConf.to_container(spec, resolve=True) for spec in configured]
        specs = []
        for f in fields:
            if f.kind not in self.registry.names():
                logger.info(
                    "No UI asset for kind %r; field %r is not shown", f.kind, f.name
                )
                continue
            specs.append({"key": f.name, "type": f.kind, "label": f.label})
        return specs

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
        converted for Gradio by :func:`to_ui`.

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
                result = self.model(**dict(zip(input_keys, values)))
                outputs = [to_ui(result[key]) for key in output_keys]
                return outputs[0] if len(outputs) == 1 else outputs
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
        demo_config_path: The packed ``demo.yaml``; a relative path is taken
            from ``demo_dir``.

    Returns:
        The loaded session.

    Raises:
        FileNotFoundError: If the demo directory or the config is missing.

    Examples:
        >>> session = load_demo_session("exp/demo", "demo.yaml")
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
    model = _build_demo_model(demo_cfg, demo_root)
    return DemoSession(demo_root, demo_cfg, model, DEFAULT_UI_ASSETS.clone())


def _build_demo_model(demo_cfg, demo_dir: Path) -> InferenceAPI:
    """Load the model ``demo.yaml`` names, a directory relative to the demo or a tag."""
    model_cfg = demo_cfg.model
    dir_or_tag = model_cfg.get("dir_or_tag")
    if not dir_or_tag:
        raise ValueError("demo config must contain model.dir_or_tag.")
    trust_user_code = bool(model_cfg.get("trust_user_code", False))
    device = str(model_cfg.get("device", "cpu"))
    raw_ref = str(dir_or_tag)
    candidate = Path(raw_ref).expanduser()
    resolved = (
        demo_dir / candidate if not candidate.is_absolute() else candidate
    ).resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise FileNotFoundError(
                "model.dir_or_tag resolved to a filesystem path, but it is not "
                f"a directory: {resolved}"
            )
        target: str | Path = resolved
    else:
        target = raw_ref
    logger.info(
        "Loading demo model | dir_or_tag=%s device=%s trust_user_code=%s",
        target,
        device,
        trust_user_code,
    )
    return load(target, device=device, trust_user_code=trust_user_code)
