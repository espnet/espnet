from __future__ import annotations

from pathlib import Path

import pytest

from espnet3.publication.demo.assets import build_runtime_overrides, load_demo_session
from espnet3.publication.demo.demo import build_demo


class DummyProvider:
    @staticmethod
    def build_model(config):
        _ = config

        def model(speech, **kwargs):
            _ = (speech, kwargs)
            return {"hyp": "ok"}

        return model


class DeviceEchoProvider:
    @staticmethod
    def build_model(config):
        device = getattr(config, "device", None)

        def model(speech, **kwargs):
            _ = kwargs
            return {"hyp": f"{device}:{speech}"}

        return model


class ModelConfigEchoProvider:
    @staticmethod
    def build_model(config):
        device = getattr(config, "device", None)
        beam_size = getattr(config.model, "beam_size", None)

        def model(speech, **kwargs):
            _ = kwargs
            return {"hyp": f"{device}:{beam_size}:{speech}"}

        return model


def _write_model_pack(
    demo_dir: Path,
    provider_target: str = "test.espnet3.demo.test_app_builder.DummyProvider",
) -> None:
    model_pack_dir = demo_dir / "model_pack"
    (model_pack_dir / "conf").mkdir(parents=True)
    (model_pack_dir / "conf" / "inference.yaml").write_text(
        "model: {}\n"
        "provider:\n"
        f"  _target_: {provider_target}\n",
        encoding="utf-8",
    )
    (model_pack_dir / "meta.yaml").write_text(
        "yaml_files:\n"
        "  inference_config: conf/inference.yaml\n",
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
        "ui:\n"
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
    (demo_dir / "packed_demo.yaml").write_text(
        "model:\n"
        "  dir_or_tag: model_pack\n"
        "ui:\n"
        "  description: \"**inline**\"\n"
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


def test_build_demo_loads_recipe_ui_registration(tmp_path: Path) -> None:
    gr = pytest.importorskip("gradio")
    _ = gr
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(demo_dir)
    (demo_dir / "custom").mkdir()
    (demo_dir / "custom" / "ui_assets.py").write_text(
        "from espnet3.publication.demo.assets import UIAsset, register_asset\n"
        "import gradio as gr\n"
        "\n"
        "class PromptTextUI(UIAsset):\n"
        "    def build_input(self, spec):\n"
        "        return gr.Textbox(label=spec.get('label', ''))\n"
        "\n"
        'register_asset("prompt_text", PromptTextUI)\n',
        encoding="utf-8",
    )
    (demo_dir / "demo_pack.yaml").write_text(
        "model:\n"
        "  dir_or_tag: model_pack\n"
        "ui:\n"
        "  asset_registry: custom/ui_assets.py\n"
        "  inputs:\n"
        "    - key: speech\n"
        "      type: prompt_text\n"
        "      label: Prompt\n"
        "  outputs:\n"
        "    - key: hyp\n"
        "      type: text\n"
        "      label: Transcription\n",
        encoding="utf-8",
    )
    app = build_demo(demo_dir)
    textboxes = _find_block_components(app, "Textbox")
    assert len(textboxes) == 2
    assert textboxes[0].label == "Prompt"


def test_custom_ui_asset_can_transform_inputs_and_outputs(tmp_path: Path) -> None:
    gr = pytest.importorskip("gradio")
    _ = gr
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(demo_dir)
    (demo_dir / "custom").mkdir()
    (demo_dir / "custom" / "ui_assets.py").write_text(
        "from espnet3.publication.demo.assets import UIAsset, register_asset\n"
        "import gradio as gr\n"
        "\n"
        "class PromptTextUI(UIAsset):\n"
        "    def build_input(self, spec):\n"
        "        return gr.Textbox(label=spec.get('label', ''), lines=3)\n"
        "\n"
        "    def normalize_input(self, value, spec):\n"
        "        _ = spec\n"
        '        return f\"normalized:{value}\"\n'
        "\n"
        "class ResultTextUI(UIAsset):\n"
        "    def build_output(self, spec):\n"
        "        return gr.Textbox(label=spec.get('label', ''), lines=2)\n"
        "\n"
        "    def format_output(self, value, spec):\n"
        "        _ = spec\n"
        '        return f\"formatted:{value}\"\n'
        "\n"
        'register_asset("prompt_text", PromptTextUI)\n'
        'register_asset("result_text", ResultTextUI)\n',
        encoding="utf-8",
    )
    (demo_dir / "demo_pack.yaml").write_text(
        "model:\n"
        "  dir_or_tag: model_pack\n"
        "inference_args:\n"
        "  beam_size: 2\n"
        "ui:\n"
        "  asset_registry: custom/ui_assets.py\n"
        "  inputs:\n"
        "    - key: speech\n"
        "      type: prompt_text\n"
        "      label: Prompt\n"
        "  outputs:\n"
        "    - key: hyp\n"
        "      type: result_text\n"
        "      label: Result\n",
        encoding="utf-8",
    )

    session = load_demo_session(demo_dir)
    input_specs = session.resolve_input_specs()
    output_specs = session.resolve_output_specs()
    input_component = session.build_input_component(input_specs[0])
    output_component = session.build_output_component(output_specs[0])
    inference_fn = session.create_inference_fn(input_specs, output_specs)

    assert input_component.__class__.__name__ == "Textbox"
    assert output_component.__class__.__name__ == "Textbox"
    assert session.inference_args == {"beam_size": 2}
    assert inference_fn("hello") == "formatted:ok"


def test_load_demo_session_applies_device_override(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(
        demo_dir,
        provider_target="test.espnet3.demo.test_app_builder.DeviceEchoProvider",
    )
    (demo_dir / "demo.yaml").write_text(
        "model:\n"
        "  dir_or_tag: model_pack\n"
        "ui:\n"
        "  inputs:\n"
        "    - key: speech\n"
        "      type: text\n"
        "      label: Input Text\n"
        "  outputs:\n"
        "    - key: hyp\n"
        "      type: text\n"
        "      label: Output Text\n",
        encoding="utf-8",
    )

    session = load_demo_session(demo_dir, model_overrides={"device": "cuda"})
    inference_fn = session.create_inference_fn(
        session.resolve_input_specs(),
        session.resolve_output_specs(),
    )

    assert session.model.resolved_device == "cuda"
    assert inference_fn("hello") == "cuda:hello"


def test_load_demo_session_logs_resolved_device(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(
        demo_dir,
        provider_target="test.espnet3.demo.test_app_builder.DeviceEchoProvider",
    )
    (demo_dir / "demo.yaml").write_text(
        "model:\n"
        "  dir_or_tag: model_pack\n",
        encoding="utf-8",
    )

    session = load_demo_session(demo_dir, model_overrides={"device": "cuda:0"})
    captured = capsys.readouterr()

    assert session.model.resolved_device == "cuda:0"
    assert "resolved_device=cuda:0" in captured.out


def test_build_runtime_overrides_supports_nested_values() -> None:
    overrides = build_runtime_overrides(
        override_args=[
            "model.beam_size=3",
            "parallel.n_workers=2",
        ],
        device="cuda:0",
    )

    assert overrides == {
        "device": "cuda:0",
        "model": {"beam_size": 3},
        "parallel": {"n_workers": 2},
    }


def test_load_demo_session_applies_nested_model_overrides(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_model_pack(
        demo_dir,
        provider_target="test.espnet3.demo.test_app_builder.ModelConfigEchoProvider",
    )
    (demo_dir / "demo.yaml").write_text(
        "model:\n"
        "  dir_or_tag: model_pack\n",
        encoding="utf-8",
    )

    session = load_demo_session(
        demo_dir,
        model_overrides={
            "device": "cuda:0",
            "model": {"beam_size": 3},
        },
    )
    inference_fn = session.create_inference_fn(
        [{"key": "speech", "type": "text"}],
        [{"key": "hyp", "type": "text"}],
    )

    assert session.model.resolved_device == "cuda:0"
    assert inference_fn("hello") == "cuda:0:3:hello"
