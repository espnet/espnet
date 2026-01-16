"""ASR demo UI builder."""

from __future__ import annotations


def build_ui_default():
    return {
        "title": "ASR Demo",
        "inputs": [
            {"name": "speech", "type": "audio", "sources": ["mic", "upload"]},
        ],
        "outputs": [
            {"name": "text", "type": "textbox"},
        ],
    }


def build_inference_default():
    return {
        "output_keys": {"text": "hyp"},
        "extra_kwargs": {},
    }


def build_ui(demo_cfg):
    title = getattr(getattr(demo_cfg, "ui", None), "title", None)
    ui = build_ui_default()
    if title:
        ui["title"] = title
    return ui
