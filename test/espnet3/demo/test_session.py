"""Demo sessions use explicit UI config and ordinary recipe callables."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from espnet3.publication.demo.assets import DEFAULT_UI_ASSETS, UIAsset
from espnet3.publication.demo.session import load_demo_session


def _config(tmp_path, **ui):
    path = tmp_path / "demo.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                # Session loading must never instantiate or inspect a model target.
                "model": {"_target_": "absent.module.Backend"},
                "ui": {
                    "title": "Recipe demo",
                    "description": None,
                    "inputs": [{"key": "prompt", "type": "text", "label": "Prompt"}],
                    "outputs": [{"key": "text", "type": "text", "label": "Result"}],
                    **ui,
                },
            }
        ),
        path,
    )
    return path


def test_session_uses_callable_without_model_declarations(tmp_path):
    class Decoder:
        @property
        def inputs(self):
            raise AssertionError("model inputs must not be inspected")

        @property
        def outputs(self):
            raise AssertionError("model outputs must not be inspected")

        def __call__(self, *, prompt):
            return {"text": prompt.upper()}

    _config(tmp_path)
    session = load_demo_session(tmp_path, "demo.yaml", model=Decoder())
    assert session.title == "Recipe demo"
    run = session.create_inference_fn(session.input_specs, session.output_specs)
    assert run("hello") == "HELLO"
    assert session.build_input_component(session.input_specs[0]).label == "Prompt"
    assert session.build_output_component(session.output_specs[0]).label == "Result"


def test_session_preserves_native_gradio_values_and_output_order(tmp_path):
    _config(tmp_path)
    audio = (16000, np.zeros(16))
    session = load_demo_session(
        tmp_path, "demo.yaml", model=lambda **values: {"audio": audio, "text": "ok"}
    )
    run = session.create_inference_fn(
        input_keys=["prompt"], output_keys=["audio", "text"]
    )
    result = run("hello")
    assert result[0] is audio
    assert result[1] == "ok"


@pytest.mark.parametrize("description", ["README.md", "Inline description"])
def test_session_loads_description_without_markdown_front_matter(tmp_path, description):
    (tmp_path / "README.md").write_text("---\ntitle: Space\n---\n\nDescription\n")
    _config(tmp_path, title=None, description=description)
    session = load_demo_session(tmp_path, "demo.yaml", model=lambda: {})
    assert session.title is None
    assert session.description == (
        "Description\n" if description == "README.md" else description
    )


@pytest.mark.parametrize("missing", ["directory", "config"])
def test_session_reports_missing_paths(tmp_path, missing):
    root = tmp_path / "absent" if missing == "directory" else tmp_path
    with pytest.raises(FileNotFoundError):
        load_demo_session(root, "demo.yaml", model=lambda: {})


def test_session_requires_explicit_output_specs(tmp_path):
    _config(tmp_path, outputs=[])
    with pytest.raises(ValueError, match="ui.outputs"):
        load_demo_session(tmp_path, "demo.yaml", model=lambda: {})


def test_session_clones_recipe_asset_registry(tmp_path):
    class PromptUI(UIAsset):
        def build_input(self, spec):
            return spec["label"]

    registry = DEFAULT_UI_ASSETS.clone()
    registry.register("prompt", PromptUI)
    _config(tmp_path, inputs=[{"key": "prompt", "type": "prompt", "label": "Custom"}])
    session = load_demo_session(
        tmp_path, "demo.yaml", model=lambda: {}, registry=registry
    )
    assert session.build_input_component(session.input_specs[0]) == "Custom"
    session.registry.register("another", PromptUI)
    assert "another" not in registry.names()
    assert "prompt" not in DEFAULT_UI_ASSETS.names()


def test_session_preserves_callable_errors(tmp_path):
    def decode(**values):
        raise RuntimeError("backend failed")

    _config(tmp_path)
    session = load_demo_session(tmp_path, "demo.yaml", model=decode)
    run = session.create_inference_fn(session.input_specs, session.output_specs)
    with pytest.raises(RuntimeError, match="backend failed"):
        run("hello")
