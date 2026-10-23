"""Tests for the codec TEMPLATE Gradio launcher's output adapter."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gradio")

from egs3.TEMPLATE.esp2_codec.src import app as codec_app  # noqa: E402

AUDIO_INPUT_SPECS = [{"key": "audio", "type": "audio", "label": "Input audio"}]
AUDIO_OUTPUT_SPECS = [{"key": "wav", "type": "audio", "label": "Resynthesized audio"}]
TEXT_INPUT_SPECS = [{"key": "text", "type": "text", "label": "Text"}]
TEXT_OUTPUT_SPECS = [{"key": "text", "type": "text", "label": "Text"}]


def test_convert_demo_audio_returns_rate_and_int16_without_normalizing():
    """A quiet signal keeps its level: Gradio would peak-normalize a float."""
    quiet = np.full(240, 0.5, dtype=np.float32)

    sample_rate, samples = codec_app.convert_demo_audio_to_int16((24000, quiet))

    assert sample_rate == 24000
    assert samples.dtype == np.int16
    assert samples.shape == (240,)
    assert abs(int(samples[0]) - round(0.5 * 32767)) <= 1


def test_convert_demo_audio_keeps_the_channel_layout():
    """Samples-first stereo, as the session lays it out, stays samples first."""
    stereo = np.zeros((240, 2), dtype=np.float32)

    _, samples = codec_app.convert_demo_audio_to_int16((16000, stereo))

    assert samples.shape == (240, 2)


def test_convert_demo_audio_clips_out_of_range_samples():
    """Out-of-range floats clip instead of wrapping around."""
    _, samples = codec_app.convert_demo_audio_to_int16((24000, np.array([2.0, -2.0])))

    assert samples.tolist() == [32767, -32767]


def test_convert_demo_audio_rejects_values_that_are_not_pairs():
    """A bare array carries no rate and is refused."""
    with pytest.raises(TypeError, match="sample_rate, samples"):
        codec_app.convert_demo_audio_to_int16(np.zeros(10, dtype=np.float32))


def test_wrap_inference_fn_passes_inputs_through_and_converts_audio_outputs():
    """The input reaches the session untouched; the audio output becomes int16."""
    seen = {}

    def resynthesize(audio):
        """Record the input and return float audio as the session would."""
        seen["audio"] = audio
        return 24000, np.full(1200, 0.25, dtype=np.float32)

    run_inference = codec_app.wrap_inference_fn(resynthesize, AUDIO_OUTPUT_SPECS)
    given = (48000, np.zeros(2400, dtype=np.int16))

    sample_rate, samples = run_inference(given)

    assert seen["audio"] is given
    assert sample_rate == 24000
    assert samples.dtype == np.int16
    assert samples.shape == (1200,)


def test_wrap_inference_fn_leaves_text_specs_untouched():
    """A text output is returned as the session produced it."""
    run_inference = codec_app.wrap_inference_fn(
        lambda text: text.upper(), TEXT_OUTPUT_SPECS
    )

    assert run_inference("abc") == "ABC"


def test_wrap_inference_fn_handles_multiple_outputs_positionally():
    """Each output is converted by its own spec, in order."""
    output_specs = [
        {"key": "text", "type": "text", "label": "Text"},
        {"key": "wav", "type": "audio", "label": "Audio"},
    ]

    def label_audio(audio):
        """Return a text and an audio output."""
        return ["hello", (24000, np.zeros(240, dtype=np.float32))]

    run_inference = codec_app.wrap_inference_fn(label_audio, output_specs)

    text, (sample_rate, samples) = run_inference((24000, np.zeros(240)))

    assert text == "hello"
    assert sample_rate == 24000
    assert samples.dtype == np.int16


def _make_session(input_specs, output_specs, title, inference_fn):
    """Build a stand-in for ``DemoSession`` with fixed specs."""
    import gradio as gr

    class FakeSession:
        """Minimal session: specs, title and component builders."""

        description = None

        def __init__(self):
            """Expose the specs the launcher reads."""
            self.title = title
            self.input_specs = input_specs
            self.output_specs = output_specs

        def create_inference_fn(self, input_specs, output_specs):
            """Return the handler the launcher wraps."""
            return inference_fn

        def build_input_component(self, spec):
            """Build one input widget by spec type."""
            return (
                gr.Audio(label=spec["label"])
                if spec["type"] == "audio"
                else gr.Textbox(label=spec["label"])
            )

        def build_output_component(self, spec):
            """Build one output widget by spec type."""
            return (
                gr.Audio(label=spec["label"])
                if spec["type"] == "audio"
                else gr.Textbox(label=spec["label"])
            )

    return FakeSession()


def test_build_demo_applies_the_zerogpu_decorator_to_the_wrapped_handler(
    tmp_path, monkeypatch
):
    """On a Space the decorated handler is the one that returns int16 audio."""
    decorated = []

    class FakeSpaces:
        """Stand in for the ``spaces`` package and record what it decorates."""

        @staticmethod
        def GPU(function):
            """Record the function and return it unchanged."""
            decorated.append(function)
            return function

    session = _make_session(
        AUDIO_INPUT_SPECS,
        AUDIO_OUTPUT_SPECS,
        "codec demo",
        lambda audio: (24000, np.full(240, 0.5, dtype=np.float32)),
    )
    monkeypatch.setattr(codec_app, "load_demo_session", lambda *a, **k: session)
    monkeypatch.setattr(codec_app, "spaces", FakeSpaces)

    app = codec_app.build_demo(tmp_path)

    assert app is not None
    assert len(decorated) == 1
    sample_rate, samples = decorated[0]((24000, np.zeros(240, dtype=np.int16)))
    assert sample_rate == 24000
    assert samples.dtype == np.int16


def test_build_demo_works_without_spaces_package(tmp_path, monkeypatch):
    """Without the `spaces` package the app still builds."""
    session = _make_session(
        TEXT_INPUT_SPECS, TEXT_OUTPUT_SPECS, None, lambda text: text
    )
    monkeypatch.setattr(codec_app, "load_demo_session", lambda *a, **k: session)
    monkeypatch.setattr(codec_app, "spaces", None)

    assert codec_app.build_demo(tmp_path) is not None
