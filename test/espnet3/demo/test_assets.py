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


def test_segments_has_an_asset_so_a_timed_model_can_be_demoed() -> None:
    """A system whose contract outputs `segments`, such as s2t, needs one.

    Without it DemoSession.build_output_component raises KeyError for the
    kind, which `segments` being a registered contract kind does not prevent.
    """
    assert "segments" in demo_assets_module.DEFAULT_UI_ASSETS._assets


def test_default_segments_ui_shows_the_list_as_json(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeJSON:
        def __init__(self, **kwargs):
            calls.append(("JSON", kwargs))

    class FakeGradio:
        JSON = FakeJSON

    monkeypatch.setattr(demo_assets_module, "gr", FakeGradio)
    demo_assets_module.DefaultSegmentsUI().build_input({"label": "In"})
    demo_assets_module.DefaultSegmentsUI().build_output({"label": "Segments"})
    assert calls == [("JSON", {"label": "In"}), ("JSON", {"label": "Segments"})]
