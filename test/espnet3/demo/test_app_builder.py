"""The packed demo: its session, its app, and the wiring to an Inference."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import egs3.TEMPLATE.esp2_asr.src.app as demo_module
import espnet3.publication.demo.assets as demo_assets_module
import espnet3.publication.demo.session as demo_session_module
from egs3.TEMPLATE.esp2_asr.src.app import build_demo
from espnet3.api.inference import Audio, Field, InferenceAPI
from espnet3.publication.demo.session import DemoSession, load_demo_session, to_ui


class Transcriber(InferenceAPI):
    """What a bundle's model looks like to the demo: fields and a result."""

    inputs = (Field("speech", "audio", "Input Audio"),)
    outputs = (Field("text", "text", "Transcription"),)

    def __init__(self, device="cpu", beam_size=1):
        self.device = device
        self.beam_size = beam_size

    @classmethod
    def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
        return cls(device=device)

    sample_rate = 8000

    def run(self, speech):
        return {"text": f"beam={self.beam_size}:{speech.duration:.1f}s"}


class Enhancer(InferenceAPI):
    """Audio out, to check the Gradio conversion and the derived specs."""

    inputs = (Field("speech", "audio"), Field("note", "text", optional=True))
    outputs = (Field("speech", "audio", "Enhanced"), Field("segments", "segments"))

    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
        return cls()

    sample_rate = 8000

    def run(self, speech, note=""):
        return {
            "speech": speech,
            "segments": [{"text": note, "start": 0.0, "end": 1.0}],
        }


class Segmenter(InferenceAPI):
    """Only segments come out, a kind with no UI asset."""

    inputs = (Field("speech", "audio"),)
    outputs = (Field("segments", "segments"),)

    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
        return cls(device=device)

    sample_rate = 8000

    def run(self, speech):
        return {"segments": []}


def _write_model_pack(
    demo_dir: Path, target: str = f"{__name__}.Transcriber", **model
) -> None:
    model_pack_dir = demo_dir / "model_pack"
    (model_pack_dir / "conf").mkdir(parents=True)
    lines = [f"model:\n  _target_: {target}\n"] + [
        f"  {k}: {v}\n" for k, v in model.items()
    ]
    (model_pack_dir / "conf" / "inference.yaml").write_text(
        "".join(lines), encoding="utf-8"
    )
    (model_pack_dir / "meta.yaml").write_text(
        "schema_version: 1\nyaml_files:\n  inference_config: conf/inference.yaml\n",
        encoding="utf-8",
    )


def _write_demo(demo_dir: Path, ui: str = "", model: str = "") -> None:
    (demo_dir / "demo.yaml").write_text(
        "model:\n  dir_or_tag: model_pack\n  trust_user_code: false\n"
        + model
        + "ui:\n"
        + ui,
        encoding="utf-8",
    )


def _find_block_components(app, component_name: str) -> list[object]:
    return [
        block for block in app.blocks.values() if type(block).__name__ == component_name
    ]


def test_specs_come_from_the_declaration_unless_the_demo_says_otherwise(tmp_path):
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(demo_dir)
    _write_demo(demo_dir, ui="  title: null\n  description: null\n")
    session = load_demo_session(demo_dir, demo_dir / "demo.yaml")
    assert session.input_specs == [
        {"key": "speech", "type": "audio", "label": "Input Audio"}
    ]
    assert session.output_specs == [
        {"key": "text", "type": "text", "label": "Transcription"}
    ]

    _write_demo(
        demo_dir,
        ui=(
            "  title: null\n  description: null\n  outputs:\n"
            "    - key: text\n      type: text\n      label: Words\n"
        ),
    )
    session = load_demo_session(demo_dir, demo_dir / "demo.yaml")
    assert session.output_specs[0]["label"] == "Words"


def test_fields_without_a_ui_asset_are_left_out(tmp_path):
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(demo_dir, target=f"{__name__}.Enhancer")
    _write_demo(demo_dir, ui="  title: null\n  description: null\n")
    session = load_demo_session(demo_dir, demo_dir / "demo.yaml")
    assert [s["key"] for s in session.input_specs] == ["speech", "note"]
    assert [s["key"] for s in session.output_specs] == [
        "speech"
    ]  # no asset for segments


def test_inference_fn_calls_the_model_by_field_and_converts_for_gradio(tmp_path):
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(demo_dir, target=f"{__name__}.Enhancer")
    _write_demo(demo_dir, ui="  title: null\n  description: null\n")
    session = load_demo_session(demo_dir, demo_dir / "demo.yaml")
    run = session.create_inference_fn(session.input_specs, session.output_specs)
    rate, samples = run((8000, np.zeros(8000, dtype=np.float32)), "hi")
    assert rate == 8000 and samples.shape == (8000,)
    both = session.create_inference_fn(
        input_keys=["speech"], output_keys=["speech", "segments"]
    )
    audio, segments = both((8000, np.zeros((2, 80), dtype=np.float32)))
    assert audio[1].shape == (
        80,
    )  # the audio field keeps channels=1: the reference channel
    assert segments[0]["end"] == 1.0


def test_to_ui_lays_multichannel_audio_out_for_gradio():
    assert to_ui("text") == "text"
    rate, samples = to_ui(Audio(np.zeros((2, 10), dtype=np.float32), 8000))
    assert rate == 8000 and samples.shape == (10, 2)


def test_model_arguments_live_in_the_bundle_not_the_demo(tmp_path):
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(demo_dir, beam_size=3)
    _write_demo(demo_dir, ui="  title: null\n  description: null\n")
    session = load_demo_session(demo_dir, demo_dir / "demo.yaml")
    run = session.create_inference_fn(input_keys=["speech"], output_keys=["text"])
    assert run((8000, np.zeros(8000, dtype=np.float32))) == "beam=3:1.0s"
    _write_demo(
        demo_dir,
        ui="  title: null\n  description: null\n",
        model="  call_args:\n    beam_size: 2\n",
    )
    with pytest.raises(TypeError, match="model.call_args"):
        load_demo_session(demo_dir, demo_dir / "demo.yaml")


def test_build_demo_shows_a_description_file_or_inline_text(tmp_path):
    pytest.importorskip("gradio")
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(demo_dir)
    _write_demo(demo_dir, ui="  title: null\n  description: README.md\n")
    (demo_dir / "README.md").write_text("# Demo\n\nDescription\n", encoding="utf-8")
    app = build_demo(demo_dir)
    markdowns = _find_block_components(app, "Markdown")
    assert len(markdowns) == 1 and markdowns[0].value == "# Demo\n\nDescription"
    _write_demo(demo_dir, ui='  title: null\n  description: "**inline**"\n')
    app = build_demo(demo_dir)
    assert _find_block_components(app, "Markdown")[0].value == "**inline**"


def test_default_assets_build_the_components_for_the_specs(tmp_path):
    pytest.importorskip("gradio")
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(demo_dir)
    _write_demo(
        demo_dir,
        ui=(
            "  title: null\n  description: null\n  inputs:\n"
            "    - key: speech\n      type: text\n      label: Prompt\n"
        ),
    )
    session = load_demo_session(demo_dir, demo_dir / "demo.yaml")
    assert (
        session.build_input_component(session.input_specs[0]).__class__.__name__
        == "Textbox"
    )
    assert (
        session.build_output_component(session.output_specs[0]).__class__.__name__
        == "Textbox"
    )


def test_build_input_component_requires_type(tmp_path):
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(demo_dir)
    _write_demo(demo_dir, ui="  title: null\n  description: null\n")
    session = load_demo_session(demo_dir, demo_dir / "demo.yaml")
    with pytest.raises(KeyError, match="type"):
        session.build_input_component({"key": "speech", "label": "Input"})


def test_demo_main_writes_demo_log(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class DummyApp:
        def launch(self) -> None:
            calls["launched"] = True

    def fake_configure_logging(log_dir: Path, filename: str):
        calls["log_dir"] = log_dir
        calls["filename"] = filename

    def fake_build_demo(
        demo_dir: Path, demo_config_path: Path | None = None
    ) -> DummyApp:
        calls["demo_dir"] = demo_dir
        calls["demo_config_path"] = demo_config_path
        return DummyApp()

    monkeypatch.setattr(demo_module, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(demo_module, "build_demo", fake_build_demo)
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: Namespace(demo_dir=tmp_path, demo_config=None),
    )
    demo_module.main()
    assert calls["log_dir"] == tmp_path and calls["filename"] == "demo.log"
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
        {"label": "Input Audio", "args": {"type": "filepath"}}
    )
    demo_assets_module.DefaultAudioUI().build_output({"label": "Output Audio"})
    demo_assets_module.DefaultTextUI().build_input({"label": "Input Text"})
    demo_assets_module.DefaultTextUI().build_output({"label": "Output Text"})
    assert calls == [
        ("Audio", {"label": "Input Audio"}),
        ("Audio", {"label": "Output Audio"}),
        ("Textbox", {"label": "Input Text"}),
        ("Textbox", {"label": "Output Text"}),
    ]


def test_build_demo_model_loads_a_tag_through_load(monkeypatch, tmp_path: Path) -> None:
    demo_cfg = OmegaConf.create(
        {
            "model": {
                "dir_or_tag": "espnet/test-model",
                "trust_user_code": True,
                "device": "cuda:0",
            },
            "ui": {"app_script": "src/app.py", "title": None, "description": None},
        }
    )
    captured = {}

    def fake_load(target, *, device="cpu", trust_user_code=False, **kwargs):
        captured.update(target=target, device=device, trust=trust_user_code)
        return Transcriber(device=device)

    monkeypatch.setattr(demo_session_module, "load", fake_load)
    model = demo_session_module._build_demo_model(demo_cfg, tmp_path)
    assert isinstance(model, Transcriber)
    assert captured == {
        "target": "espnet/test-model",
        "device": "cuda:0",
        "trust": True,
    }
    # a directory next to the demo is resolved before it is handed over
    (tmp_path / "model_pack").mkdir()
    demo_cfg.model.dir_or_tag = "model_pack"
    demo_session_module._build_demo_model(demo_cfg, tmp_path)
    assert captured["target"] == (tmp_path / "model_pack").resolve()
    session = DemoSession(
        tmp_path, demo_cfg, model, demo_assets_module.DEFAULT_UI_ASSETS.clone()
    )
    assert session.input_specs[0]["key"] == "speech"


def test_a_demo_with_nothing_to_show_says_so(tmp_path):
    """Not an app that runs the model and shows nothing."""
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(demo_dir, target=f"{__name__}.Segmenter")
    _write_demo(demo_dir, ui="  title: null\n  description: null\n")
    with pytest.raises(
        ValueError, match=r"no UI asset is registered for \['segments'\]"
    ):
        load_demo_session(demo_dir, demo_dir / "demo.yaml")
