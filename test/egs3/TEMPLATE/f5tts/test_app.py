"""Tests for the helpers of the template's demo launcher."""

from types import SimpleNamespace

import numpy as np
import pytest

from espnet3.api.inference import Audio

pytest.importorskip("gradio")

from egs3.TEMPLATE.f5tts.src.app import (  # noqa: E402
    build_gradio_audio,
    resolve_sample_rate,
)


def test_resolve_sample_rate_reads_the_contract_class() -> None:
    assert resolve_sample_rate(SimpleNamespace(sample_rate=24000)) == 24000


def test_resolve_sample_rate_reads_the_bare_engine() -> None:
    assert resolve_sample_rate(SimpleNamespace(target_sample_rate=22050)) == 22050


def test_resolve_sample_rate_refuses_a_model_without_a_rate() -> None:
    with pytest.raises(TypeError, match="output sample rate"):
        resolve_sample_rate(SimpleNamespace())


def test_build_gradio_audio_pairs_an_array_with_the_given_rate() -> None:
    rate, samples = build_gradio_audio(np.zeros(4, dtype=np.float64), 24000)

    assert rate == 24000
    assert samples.dtype == np.float32
    assert samples.shape == (4,)


def test_build_gradio_audio_uses_the_rate_an_audio_carries() -> None:
    audio = Audio(np.ones(4, dtype=np.float32), 16000)

    rate, samples = build_gradio_audio(audio, 24000)

    assert rate == 16000
    np.testing.assert_array_equal(samples, audio.array)
