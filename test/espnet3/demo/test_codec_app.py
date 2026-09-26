"""Tests for the codec TEMPLATE Gradio launcher's audio adapters."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gradio")

from egs3.TEMPLATE.esp2_codec.src import app as codec_app  # noqa: E402

AUDIO_INPUT_SPECS = [{"key": "audio", "type": "audio", "label": "Input Audio"}]
AUDIO_OUTPUT_SPECS = [{"key": "wav", "type": "audio", "label": "Resynthesized Audio"}]
TEXT_INPUT_SPECS = [{"key": "text", "type": "text", "label": "Text"}]
TEXT_OUTPUT_SPECS = [{"key": "text", "type": "text", "label": "Text"}]


def _make_sine_wave(
    sample_rate: int, seconds: float = 1.0, frequency: float = 440.0
) -> np.ndarray:
    times = np.arange(int(sample_rate * seconds)) / sample_rate
    return np.sin(2 * np.pi * frequency * times).astype(np.float32)


def test_convert_gradio_audio_converts_int16_stereo_and_resamples():
    sample_rate = 48000
    mono = _make_sine_wave(sample_rate)
    stereo_int16 = (np.stack([mono, mono], axis=1) * 32767).astype(np.int16)

    waveform = codec_app.convert_gradio_audio_to_waveform(
        (sample_rate, stereo_int16), sample_rate=24000
    )

    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert abs(waveform.shape[0] - 24000) <= 1
    assert 0.9 < float(np.abs(waveform).max()) <= 1.0


def test_convert_gradio_audio_passes_float_mono_at_target_rate_through():
    expected = _make_sine_wave(24000)

    waveform = codec_app.convert_gradio_audio_to_waveform(
        (24000, expected), sample_rate=24000
    )

    np.testing.assert_allclose(waveform, expected)


def test_convert_gradio_audio_rejects_values_that_are_not_tuples():
    with pytest.raises(TypeError, match="sample_rate, samples"):
        codec_app.convert_gradio_audio_to_waveform(
            np.zeros(10, dtype=np.float32), sample_rate=24000
        )


def test_convert_gradio_audio_keeps_none_for_empty_component():
    assert codec_app.convert_gradio_audio_to_waveform(None, sample_rate=24000) is None


def test_convert_waveform_returns_rate_and_flat_int16_without_normalizing():
    quiet = np.full((1, 240), 0.5, dtype=np.float64)

    sample_rate, samples = codec_app.convert_waveform_to_gradio_audio(quiet, 24000)

    assert sample_rate == 24000
    assert samples.dtype == np.int16
    assert samples.shape == (240,)
    # Gradio peak-normalizes float arrays; int16 keeps the codec's level.
    assert abs(int(samples[0]) - round(0.5 * 32767)) <= 1


def test_convert_waveform_clips_out_of_range_samples():
    _, samples = codec_app.convert_waveform_to_gradio_audio(
        np.array([2.0, -2.0]), 24000
    )

    assert samples.tolist() == [32767, -32767]


def test_wrap_inference_fn_converts_audio_input_and_output():
    seen = {}

    def halve_audio(audio):
        seen["audio"] = audio
        return audio[::2]

    run_inference = codec_app.wrap_inference_fn(
        halve_audio, AUDIO_INPUT_SPECS, AUDIO_OUTPUT_SPECS, sample_rate=24000
    )
    samples_int16 = np.full(2400, 0.5 * 32767, dtype=np.int16)

    sample_rate, samples = run_inference((24000, samples_int16))

    assert seen["audio"].dtype == np.float32
    assert seen["audio"].shape == (2400,)
    assert sample_rate == 24000
    assert samples.shape == (1200,)


def test_wrap_inference_fn_requires_sample_rate_for_audio_specs():
    with pytest.raises(ValueError, match="ui.sample_rate"):
        codec_app.wrap_inference_fn(
            lambda value: value,
            AUDIO_INPUT_SPECS,
            AUDIO_OUTPUT_SPECS,
            sample_rate=None,
        )


def test_wrap_inference_fn_leaves_text_specs_untouched():
    run_inference = codec_app.wrap_inference_fn(
        lambda text: text.upper(),
        TEXT_INPUT_SPECS,
        TEXT_OUTPUT_SPECS,
        sample_rate=None,
    )

    assert run_inference("abc") == "ABC"


def test_wrap_inference_fn_handles_multiple_outputs_positionally():
    output_specs = [
        {"key": "text", "type": "text", "label": "Text"},
        {"key": "wav", "type": "audio", "label": "Audio"},
    ]

    def label_audio(audio):
        return ["hello", audio]

    run_inference = codec_app.wrap_inference_fn(
        label_audio, AUDIO_INPUT_SPECS, output_specs, sample_rate=24000
    )

    text, (sample_rate, samples) = run_inference((24000, np.zeros(240, dtype=np.int16)))

    assert text == "hello"
    assert sample_rate == 24000
    assert samples.shape == (240,)


def test_build_demo_reads_sample_rate_and_applies_zerogpu_decorator(
    tmp_path, monkeypatch
):
    from omegaconf import OmegaConf

    class FakeSession:
        title = "codec demo"
        description = None
        input_specs = AUDIO_INPUT_SPECS
        output_specs = AUDIO_OUTPUT_SPECS
        demo_cfg = OmegaConf.create({"ui": {"sample_rate": 24000}})

        def create_inference_fn(self, input_specs, output_specs):
            return lambda audio: audio

        def build_input_component(self, spec):
            import gradio as gr

            return gr.Audio(label=spec["label"])

        def build_output_component(self, spec):
            import gradio as gr

            return gr.Audio(label=spec["label"])

    decorated = []

    class FakeSpaces:
        @staticmethod
        def GPU(function):
            decorated.append(function)
            return function

    monkeypatch.setattr(
        codec_app, "load_demo_session", lambda *args, **kwargs: FakeSession()
    )
    monkeypatch.setattr(codec_app, "spaces", FakeSpaces)

    app = codec_app.build_demo(tmp_path)

    assert app is not None
    assert len(decorated) == 1
    assert decorated[0]((24000, np.zeros(240, dtype=np.int16)))[0] == 24000


def test_build_demo_works_without_spaces_package(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    class FakeSession:
        title = None
        description = None
        input_specs = TEXT_INPUT_SPECS
        output_specs = TEXT_OUTPUT_SPECS
        demo_cfg = OmegaConf.create({"ui": {}})

        def create_inference_fn(self, input_specs, output_specs):
            return lambda text: text

        def build_input_component(self, spec):
            import gradio as gr

            return gr.Textbox(label=spec["label"])

        def build_output_component(self, spec):
            import gradio as gr

            return gr.Textbox(label=spec["label"])

    monkeypatch.setattr(
        codec_app, "load_demo_session", lambda *args, **kwargs: FakeSession()
    )
    monkeypatch.setattr(codec_app, "spaces", None)

    assert codec_app.build_demo(tmp_path) is not None
