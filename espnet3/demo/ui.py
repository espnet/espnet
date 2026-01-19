"""Gradio UI builders for demo configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import gradio as gr


@dataclass(frozen=True)
class UiSpec:
    """Typed UI spec extracted from demo config."""

    inputs: List[Any]
    outputs: List[Any]
    input_names: List[str]
    output_names: List[str]
    title: str | None = None
    description: str | None = None
    article: str | None = None
    button_label: str | None = None


_AUDIO_SOURCE_MAP = {
    "mic": "microphone",
    "microphone": "microphone",
    "upload": "upload",
    "file": "upload",
}


def build_ui_from_config(ui_cfg) -> UiSpec:
    """Build a UI spec from the demo UI configuration."""
    inputs_cfg = list(getattr(ui_cfg, "inputs", []) or [])
    outputs_cfg = list(getattr(ui_cfg, "outputs", []) or [])
    inputs, input_names = _build_components(inputs_cfg, is_input=True)
    outputs, output_names = _build_components(outputs_cfg, is_input=False)
    return UiSpec(
        inputs=inputs,
        outputs=outputs,
        input_names=input_names,
        output_names=output_names,
        title=getattr(ui_cfg, "title", None),
        description=getattr(ui_cfg, "description", None),
        article=getattr(ui_cfg, "article", None),
        button_label=getattr(getattr(ui_cfg, "button", None), "label", None),
    )


def _build_components(
    cfgs: Iterable[Any], *, is_input: bool
) -> Tuple[List[Any], List[str]]:
    """Build Gradio components and their names from config entries.

    Args:
        cfgs: Iterable of UI config entries, each requiring a ``name`` and ``type``.
        is_input: Flag reserved for future input/output-specific behavior.
    Returns:
        Tuple of (components, names) in the same order as cfgs.
    """
    components = []
    names = []
    for cfg in cfgs:
        name = str(getattr(cfg, "name", ""))
        if not name:
            raise ValueError("UI component must include a non-empty 'name'.")
        comp = _build_component(cfg, is_input=is_input)
        components.append(comp)
        names.append(name)
    return components, names


def _build_component(cfg, *, is_input: bool | None = None):
    """Instantiate a single Gradio component from a config entry.

    Supported ``type`` values and corresponding components:
      - ``audio``: gr.Audio (supports ``sources`` and ``audio_type``)
      - ``textbox``: gr.Textbox (supports ``lines`` and ``placeholder``)
      - ``dropdown``: gr.Dropdown (supports ``choices`` and ``value``)
      - ``number``: gr.Number (supports ``value``)
      - ``slider``: gr.Slider (supports ``min``, ``max``, ``step``, ``value``)
      - ``checkbox``: gr.Checkbox (supports ``value``)
      - ``image``: gr.Image
      - ``file``: gr.File

    Args:
        cfg: UI config entry with ``name``/``type`` and optional fields above.
        is_input: Flag reserved for future input/output-specific behavior.
    Returns:
        Gradio component instance.
    Raises:
        ValueError: For unsupported component types.
    """
    comp_type = str(getattr(cfg, "type", "")).lower()
    label = getattr(cfg, "label", None) or getattr(cfg, "name", None)
    if comp_type == "audio":
        sources = _normalize_audio_sources(getattr(cfg, "sources", None))
        return gr.Audio(
            label=label,
            sources=sources,
            type=str(getattr(cfg, "audio_type", "numpy")),
        )
    if comp_type == "textbox":
        return gr.Textbox(
            label=label,
            lines=int(getattr(cfg, "lines", 1) or 1),
            placeholder=getattr(cfg, "placeholder", None),
        )
    if comp_type == "dropdown":
        choices = list(getattr(cfg, "choices", []) or [])
        return gr.Dropdown(
            label=label, choices=choices, value=getattr(cfg, "value", None)
        )
    if comp_type == "number":
        return gr.Number(label=label, value=getattr(cfg, "value", None))
    if comp_type == "slider":
        return gr.Slider(
            minimum=float(getattr(cfg, "min", 0)),
            maximum=float(getattr(cfg, "max", 1)),
            step=float(getattr(cfg, "step", 1)),
            value=getattr(cfg, "value", None),
            label=label,
        )
    if comp_type == "checkbox":
        return gr.Checkbox(label=label, value=bool(getattr(cfg, "value", False)))
    if comp_type == "image":
        return gr.Image(label=label)
    if comp_type == "file":
        return gr.File(label=label)
    raise ValueError(f"Unsupported UI component type: {comp_type}")


def _normalize_audio_sources(sources) -> List[str]:
    if sources is None:
        return ["microphone", "upload"]
    normalized = []
    for item in sources:
        key = str(item).lower()
        normalized.append(_AUDIO_SOURCE_MAP.get(key, key))
    return normalized
