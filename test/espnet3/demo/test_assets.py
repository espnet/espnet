"""UI assets remain independent of model inference declarations."""

import pytest

import espnet3.publication.demo.assets as demo_assets_module


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
