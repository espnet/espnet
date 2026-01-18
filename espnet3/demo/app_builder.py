"""Gradio demo app launcher."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import gradio as gr
from omegaconf import DictConfig, OmegaConf

from espnet3.demo.resolve import load_demo_config
from espnet3.demo.runtime import build_runtime, run_inference
from espnet3.demo.ui import UiSpec, build_ui_from_config


def build_demo_app(demo_dir: Path) -> gr.Blocks:
    demo_cfg = load_demo_config(demo_dir)
    runtime = build_runtime(demo_cfg, demo_dir)
    ui = _load_ui_spec(demo_cfg, demo_dir)

    title = ui.title or "ESPnet Demo"
    description = ui.description
    article = ui.article
    article_path = getattr(getattr(demo_cfg, "ui", None), "article_path", None)
    if article_path:
        article = _read_article(demo_dir, str(article_path)) or article
    button_label = ui.button_label or "Run"

    with gr.Blocks(title=title) as app:
        gr.Markdown(f"# {title}")
        if description:
            gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                for comp in ui.inputs:
                    comp.render()
                button = gr.Button(button_label)
            with gr.Column():
                for comp in ui.outputs:
                    comp.render()
        if article:
            gr.Markdown(article)

        all_inputs = ui.inputs
        all_names = ui.input_names

        def _predict(*values: Any):
            return run_inference(
                runtime,
                ui_names=all_names,
                ui_values=list(values),
                output_names=ui.output_names,
            )

        button.click(fn=_predict, inputs=all_inputs, outputs=ui.outputs)
    return app


def _load_ui_spec(demo_cfg, demo_dir: Path) -> UiSpec:
    ui_cfg = getattr(demo_cfg, "ui", None)
    if ui_cfg is None:
        ui = _load_system_ui(demo_cfg, defaults=False)
        if ui is None:
            raise ValueError(
                "demo.yaml must include ui configuration or a supported system."
            )
        return _normalize_ui_spec(ui)
    defaults = _load_system_ui(demo_cfg, defaults=True)
    if defaults is None:
        return build_ui_from_config(ui_cfg)
    base = OmegaConf.create(defaults)
    return build_ui_from_config(OmegaConf.merge(base, ui_cfg))


def _normalize_ui_spec(ui) -> UiSpec:
    if isinstance(ui, UiSpec):
        return ui
    if isinstance(ui, dict):
        cfg = OmegaConf.create(ui)
        return build_ui_from_config(cfg)
    if isinstance(ui, DictConfig):
        return build_ui_from_config(ui)
    return ui


def _load_system_ui(demo_cfg, *, defaults: bool):
    system = str(getattr(demo_cfg, "system", "")).lower()
    if not system:
        return None
    module_path = f"espnet3.systems.{system}.demo"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        return None
    fn_name = "build_ui_default" if defaults else "build_ui"
    fn = getattr(module, fn_name, None)
    if fn is None:
        return None
    if defaults:
        return fn()
    return fn(demo_cfg)


def _read_article(demo_dir: Path, article_path: str) -> str | None:
    path = Path(article_path)
    if not path.is_absolute():
        path = demo_dir / path
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None
