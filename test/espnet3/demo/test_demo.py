from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from espnet3.demo.runtime import DemoRuntime, run_inference


def test_run_inference_with_output_keys() -> None:
    class DummyRunner:
        @staticmethod
        def forward(idx, *, dataset, model=None, **kwargs):
            _ = (idx, dataset, model, kwargs)
            return {"idx": 0, "hyp": "hello", "ref": ""}

    runtime = DemoRuntime(
        infer_config=None,
        model=None,
        runner_cls=DummyRunner,
        output_keys={"text": "hyp"},
        extra_kwargs={},
    )
    outputs = run_inference(
        runtime,
        ui_names=["speech"],
        ui_values=[(16000, np.array([0.0], dtype=np.float32))],
        output_names=["text"],
    )
    assert outputs == ["hello"]

def test_build_ui_from_config_smoke() -> None:
    gr = pytest.importorskip("gradio")
    from espnet3.demo.ui import build_ui_from_config

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
    assert isinstance(ui.inputs[0], gr.Audio)


def test_build_ui_from_module() -> None:
    gr = pytest.importorskip("gradio")
    from espnet3.demo.app_builder import _normalize_ui_spec
    from espnet3.systems.asr import demo as asr_demo

    ui = _normalize_ui_spec(asr_demo.build_ui({}))
    assert ui.input_names == ["speech"]
    assert ui.output_names == ["text"]
    assert isinstance(ui.inputs[0], gr.Audio)
