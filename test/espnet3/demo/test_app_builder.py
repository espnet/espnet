from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import egs3.TEMPLATE.asr.src.app as demo_module
import espnet3.publication.demo.assets as demo_assets_module
import espnet3.publication.demo.session as demo_session_module
from egs3.TEMPLATE.asr.src.app import build_demo
from espnet3.publication.demo.session import load_demo_session


class DummyProvider:
    @staticmethod
    def build_model(config):
        _ = config

        def model(speech, **kwargs):
            _ = (speech, kwargs)
            return {"hyp": "ok"}

        return model


class CallArgsEchoProvider:
    @staticmethod
    def build_model(config):
        _ = config

        def model(speech, **kwargs):
            return {"hyp": f"beam={kwargs.get('beam_size')}:{speech}"}

        return model


def _write_model_pack(
    demo_dir: Path,
    provider_target: str = "test.espnet3.demo.test_app_builder.DummyProvider",
) -> None:
    model_pack_dir = demo_dir / "model_pack"
    (model_pack_dir / "conf").mkdir(parents=True)
    (model_pack_dir / "conf" / "inference.yaml").write_text(
        "model: {}\n" "provider:\n" f"  _target_: {provider_target}\n",
        encoding="utf-8",
    )
    (model_pack_dir / "meta.yaml").write_text(
        "yaml_files:\n" "  inference_config: conf/inference.yaml\n",
        encoding="utf-8",
    )


def _find_block_components(app, component_name: str) -> list[object]:
    return [
        block for block in app.blocks.values() if type(block).__name__ == component_name
    ]


def test_build_demo_with_custom_provider(tmp_path: Path) -> None:
    gr = pytest.importorskip("gradio")
    _ = gr
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(demo_dir)
    (demo_dir / "demo.yaml").write_text(
        "model:\n"
        "  dir_or_tag: model_pack\n"
        "  trust_user_code: false\n"
        "  call_args: {}\n"
        "ui:\n"
        "  title: null\n"
        "  description: README.md\n"
        "  inputs:\n"
        "    - key: speech\n"
        "      type: audio\n"
        "      label: Input Audio\n"
        "  outputs:\n"
        "    - key: hyp\n"
        "      type: text\n"
        "      label: Transcription\n",
        encoding="utf-8",
    )
    (demo_dir / "README.md").write_text("# Demo\n\nDescription\n", encoding="utf-8")
    app = build_demo(demo_dir)
    assert app is not None
    markdowns = _find_block_components(app, "Markdown")
    assert len(markdowns) == 1
    assert markdowns[0].value == "# Demo\n\nDescription"


def test_build_demo_uses_inline_description(tmp_path: Path) -> None:
    gr = pytest.importorskip("gradio")
    _ = gr
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(demo_dir)
    (demo_dir / "demo.yaml").write_text(
        "model:\n"
        "  dir_or_tag: model_pack\n"
        "  trust_user_code: false\n"
        "  call_args: {}\n"
        "ui:\n"
        "  title: null\n"
        '  description: "**inline**"\n'
        "  inputs:\n"
        "    - key: speech\n"
        "      type: audio\n"
        "      label: Input Audio\n"
        "  outputs:\n"
        "    - key: hyp\n"
        "      type: text\n"
        "      label: Transcription\n",
        encoding="utf-8",
    )
    app = build_demo(demo_dir)
    markdowns = _find_block_components(app, "Markdown")
    assert len(markdowns) == 1
    assert markdowns[0].value == "**inline**"


def test_default_assets_pass_through_inputs_and_outputs(tmp_path: Path) -> None:
    gr = pytest.importorskip("gradio")
    _ = gr
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(
        demo_dir,
        provider_target="test.espnet3.demo.test_app_builder.CallArgsEchoProvider",
    )
    (demo_dir / "demo.yaml").write_text(
        "model:\n"
        "  dir_or_tag: model_pack\n"
        "  trust_user_code: false\n"
        "  call_args:\n"
        "    beam_size: 2\n"
        "ui:\n"
        "  title: null\n"
        "  description: null\n"
        "  inputs:\n"
        "    - key: speech\n"
        "      type: text\n"
        "      label: Prompt\n"
        "  outputs:\n"
        "    - key: hyp\n"
        "      type: text\n"
        "      label: Result\n",
        encoding="utf-8",
    )

    session = load_demo_session(demo_dir, demo_dir / "demo.yaml")
    input_specs = session.input_specs
    output_specs = session.output_specs
    input_component = session.build_input_component(input_specs[0])
    output_component = session.build_output_component(output_specs[0])
    inference_fn = session.create_inference_fn(input_specs, output_specs)

    assert input_component.__class__.__name__ == "Textbox"
    assert output_component.__class__.__name__ == "Textbox"
    assert session.call_args == {"beam_size": 2}
    assert inference_fn("hello") == "beam=2:hello"


def test_create_inference_fn_accepts_explicit_key_lists(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(
        demo_dir,
        provider_target="test.espnet3.demo.test_app_builder.CallArgsEchoProvider",
    )
    (demo_dir / "demo.yaml").write_text(
        "model:\n"
        "  dir_or_tag: model_pack\n"
        "  trust_user_code: false\n"
        "  call_args:\n"
        "    beam_size: 3\n"
        "ui:\n"
        "  title: null\n"
        "  description: null\n"
        "  inputs: []\n"
        "  outputs: []\n",
        encoding="utf-8",
    )

    session = load_demo_session(demo_dir, demo_dir / "demo.yaml")
    inference_fn = session.create_inference_fn(
        input_keys=["speech"],
        output_keys=["hyp"],
    )

    assert inference_fn("hello") == "beam=3:hello"


def test_demo_main_writes_demo_log(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class DummyApp:
        def launch(self) -> None:
            calls["launched"] = True

    def fake_configure_logging(log_dir: Path, filename: str):
        calls["log_dir"] = log_dir
        calls["filename"] = filename
        return None

    def fake_build_demo(
        demo_dir: Path,
        demo_config_path: Path | None = None,
    ) -> DummyApp:
        calls["demo_dir"] = demo_dir
        calls["demo_config_path"] = demo_config_path
        return DummyApp()

    monkeypatch.setattr(demo_module, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(demo_module, "build_demo", fake_build_demo)
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: Namespace(
            demo_dir=tmp_path,
            demo_config=None,
        ),
    )

    demo_module.main()

    assert calls["log_dir"] == tmp_path
    assert calls["filename"] == "demo.log"
    assert calls["demo_dir"] == tmp_path
    assert calls["demo_config_path"] == tmp_path / "demo.yaml"
    assert calls["launched"] is True


def test_default_text_ui_requires_gradio(monkeypatch) -> None:
    monkeypatch.setattr(demo_assets_module, "gr", None)

    with pytest.raises(ImportError, match="gradio is required"):
        demo_assets_module.DefaultTextUI()


def test_default_assets_use_only_label(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeAudio:
        def __init__(self, **kwargs):
            calls.append(("Audio", kwargs))

    class FakeTextbox:
        def __init__(self, **kwargs):
            calls.append(("Textbox", kwargs))

    class FakeGradio:
        Audio = FakeAudio
        Textbox = FakeTextbox

    monkeypatch.setattr(demo_assets_module, "gr", FakeGradio)

    demo_assets_module.DefaultAudioUI().build_input(
        {"label": "Input Audio", "args": {"type": "filepath", "sources": ["upload"]}}
    )
    demo_assets_module.DefaultAudioUI().build_output(
        {"label": "Output Audio", "args": {"format": "wav"}}
    )
    demo_assets_module.DefaultTextUI().build_input(
        {"label": "Input Text", "args": {"lines": 7}}
    )
    demo_assets_module.DefaultTextUI().build_output(
        {"label": "Output Text", "args": {"lines": 9}}
    )

    assert calls == [
        ("Audio", {"label": "Input Audio"}),
        ("Audio", {"label": "Output Audio"}),
        ("Textbox", {"label": "Input Text"}),
        ("Textbox", {"label": "Output Text"}),
    ]


def test_build_input_component_requires_type(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(demo_dir)
    (demo_dir / "demo.yaml").write_text(
        "model:\n"
        "  dir_or_tag: model_pack\n"
        "  trust_user_code: false\n"
        "  call_args: {}\n"
        "ui:\n"
        "  title: null\n"
        "  description: null\n"
        "  inputs:\n"
        "    - key: speech\n"
        "      type: audio\n"
        "      label: Input Audio\n"
        "  outputs:\n"
        "    - key: hyp\n"
        "      type: text\n"
        "      label: Transcription\n",
        encoding="utf-8",
    )

    session = load_demo_session(demo_dir, demo_dir / "demo.yaml")

    with pytest.raises(KeyError, match="type"):
        session.build_input_component({"key": "speech", "label": "Input"})


def test_build_demo_model_loads_pretrained_tag(monkeypatch, tmp_path: Path) -> None:
    demo_cfg = OmegaConf.create(
        {
            "model": {
                "dir_or_tag": "espnet/test-model",
                "trust_user_code": True,
                "call_args": {},
            },
            "ui": {
                "app_script": "src/app.py",
                "title": None,
                "description": None,
                "inputs": [{"key": "speech", "type": "audio", "label": "Input Audio"}],
                "outputs": [{"key": "hyp", "type": "text", "label": "Transcription"}],
            },
        }
    )
    captured = {}

    class DummyModel:
        model = None

    def fake_from_pretrained(model_tag, trust_user_code=False):
        captured["model_tag"] = model_tag
        captured["trust_user_code"] = trust_user_code
        return DummyModel()

    monkeypatch.setattr(
        demo_session_module.InferenceModel,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    model = demo_session_module._build_demo_model(
        demo_cfg,
        tmp_path,
    )

    assert isinstance(model, DummyModel)
    assert captured["model_tag"] == "espnet/test-model"
    assert captured["trust_user_code"] is True
