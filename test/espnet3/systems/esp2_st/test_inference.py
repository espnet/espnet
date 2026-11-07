"""The ST system's runner hooks and its short-utterance backend adapter."""

from __future__ import annotations

import numpy as np
import pytest

from espnet2.bin.st_inference import Speech2Text
from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    TooShortUttError,
)
from espnet3.contract.kinds.audio import Audio
from espnet3.systems.esp2_st.src.backend import TOO_SHORT_TEXT, Speech2TextShortSafe
from espnet3.systems.esp2_st.src.inference_runner import ESP2STInferenceRunner

NBEST = [("hallo welt", ["hallo", "welt"], [1, 2], None)]


def audio(seconds=1.0, rate=16000):
    return Audio(np.zeros(int(seconds * rate), dtype=np.float32), rate)


# -- the runner's two hooks --------------------------------------------------


def test_the_backend_is_handed_the_sample_array():
    out = ESP2STInferenceRunner.prepare_model_inputs({"speech": audio()})
    assert isinstance(out["speech"], np.ndarray)
    assert out["speech"].shape == (16000,)


def test_a_batch_is_refused_with_the_setting_that_causes_it():
    """ESPnet2 ST is @typechecked on one array and has no batch_decode."""
    with pytest.raises(ValueError, match="batch_size: null"):
        ESP2STInferenceRunner.prepare_model_inputs({"speech": [audio(), audio()]})


def test_the_best_translation_is_the_named_output():
    assert ESP2STInferenceRunner.format_output(NBEST, batched=False) == {
        "text": "hallo welt"
    }


def test_a_multidecoder_result_gives_its_translation_not_its_transcript():
    """Speech2Text returns (translations, transcripts) for those models."""
    transcripts = [("hello world", ["hello"], [3], None)]
    out = ESP2STInferenceRunner.format_output((NBEST, transcripts), batched=False)
    assert out == {"text": "hallo welt"}


def test_a_hypothesis_with_no_text_is_an_empty_string():
    """Speech2Text leaves the text None when it has no tokenizer."""
    assert ESP2STInferenceRunner.format_output(
        [(None, [], [], None)], batched=False
    ) == {"text": ""}


# -- the backend adapter -----------------------------------------------------


def test_audio_too_short_to_subsample_is_scored_as_a_blank_translation(monkeypatch):
    """egs2 keeps the row so its reference still counts against BLEU."""

    def too_short(self, *args, **kwargs):
        raise TooShortUttError("has 1 frames, but needs more than 7", 1, 7)

    monkeypatch.setattr(Speech2Text, "__call__", too_short)
    model = Speech2TextShortSafe.__new__(Speech2TextShortSafe)

    result = model(np.zeros(100, dtype=np.float32))

    assert result[0][0] == TOO_SHORT_TEXT
    assert ESP2STInferenceRunner.format_output(result, batched=False) == {
        "text": TOO_SHORT_TEXT
    }


def test_the_adapter_is_otherwise_the_backend(monkeypatch):
    """Anything that is not TooShortUttError passes straight through."""
    monkeypatch.setattr(Speech2Text, "__call__", lambda self, *a, **k: NBEST)
    model = Speech2TextShortSafe.__new__(Speech2TextShortSafe)
    assert model(np.zeros(16000, dtype=np.float32)) is NBEST

    def boom(self, *args, **kwargs):
        raise RuntimeError("something else")

    monkeypatch.setattr(Speech2Text, "__call__", boom)
    with pytest.raises(RuntimeError, match="something else"):
        model(np.zeros(16000, dtype=np.float32))
