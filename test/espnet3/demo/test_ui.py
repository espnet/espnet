from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from espnet3.demo.ui import build_ui_from_config


def test_build_ui_from_config_audio() -> None:
    gr = pytest.importorskip("gradio")
    _ = gr
    ui_cfg = OmegaConf.create(
        {
            "inputs": [
                {"name": "speech", "type": "audio", "sources": ["mic", "upload"]},
            ],
            "outputs": [
                {"name": "text", "type": "textbox"},
            ],
        }
    )
    ui = build_ui_from_config(ui_cfg)
    assert ui.input_names == ["speech"]
    assert ui.output_names == ["text"]


def test_build_ui_rejects_missing_name() -> None:
    ui_cfg = OmegaConf.create({"inputs": [{"type": "textbox"}], "outputs": []})
    with pytest.raises(ValueError, match="must include a non-empty 'name'"):
        build_ui_from_config(ui_cfg)
