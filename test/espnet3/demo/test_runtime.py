from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from espnet3.demo.runtime import DemoRuntime, run_inference


class DummyRunner:
    @staticmethod
    def forward(idx, *, dataset, model=None, **kwargs):
        _ = (idx, model, kwargs)
        audio = dataset[0]["speech"]
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        return {"hyp": "ok"}


def test_run_inference_normalizes_audio() -> None:
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
        ui_values=[(16000, np.array([0.0], dtype=np.float64))],
        output_names=["text"],
    )
    assert outputs == ["ok"]


def test_run_inference_requires_output_keys() -> None:
    runtime = DemoRuntime(
        infer_config=None,
        model=None,
        runner_cls=DummyRunner,
        output_keys={},
        extra_kwargs={},
    )
    with pytest.raises(ValueError, match="output_keys is required"):
        run_inference(
            runtime,
            ui_names=["speech"],
            ui_values=[(16000, np.array([0.0], dtype=np.float32))],
            output_names=["text"],
        )
