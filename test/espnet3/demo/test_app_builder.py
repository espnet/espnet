from __future__ import annotations

from pathlib import Path

import pytest

from espnet3.demo.app_builder import build_demo_app


class DummyProvider:
    @staticmethod
    def build_model(config):
        _ = config

        def model(speech, **kwargs):
            _ = (speech, kwargs)
            return [["ok"]]

        return model


class DummyRunner:
    @staticmethod
    def forward(idx, *, dataset, model=None, **kwargs):
        _ = (idx, kwargs)
        return {"hyp": model(dataset[0]["speech"])[0][0]}


def test_build_demo_app_with_custom_provider(tmp_path: Path) -> None:
    gr = pytest.importorskip("gradio")
    _ = gr
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    (demo_dir / "config").mkdir()
    (demo_dir / "config" / "infer.yaml").write_text(
        "model: {}\n"
        "provider:\n"
        "  _target_: test.espnet3.demo.test_app_builder.DummyProvider\n"
        "runner:\n"
        "  _target_: test.espnet3.demo.test_app_builder.DummyRunner\n",
        encoding="utf-8",
    )
    (demo_dir / "demo.yaml").write_text(
        "system: dummy\n"
        "infer_config: config/infer.yaml\n"
        "ui:\n"
        "  inputs:\n"
        "    - name: speech\n"
        "      type: audio\n"
        "  outputs:\n"
        "    - name: text\n"
        "      type: textbox\n"
        "output_keys:\n"
        "  text: hyp\n",
        encoding="utf-8",
    )
    app = build_demo_app(demo_dir)
    assert app is not None
