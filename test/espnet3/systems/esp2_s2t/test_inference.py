"""The s2t runner's output formatting and the long-form backend.

The runner only shapes what the backend returns, so the two are tested apart:
the backend against a stand-in ``Speech2Text``, the runner against the
``(start, end, text)`` tuples it is handed.
"""

from __future__ import annotations

import numpy as np
import pytest

from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    TooShortUttError,
)
from espnet3.contract.kinds.audio import Audio
from espnet3.systems.esp2_s2t.inference import ESP2S2TInferenceRunner
from espnet3.systems.esp2_s2t.src.longform_backend import S2TLongFormBackend

RATE = 16000


class _Model:
    """A stand-in for ``Speech2Text`` with just what the backend reads."""

    def __init__(self, utterances=None, tokens=None, raises=None, ctc_only=False):
        self._utterances = utterances or []
        self.s2t_model = type("M", (), {"token_list": tokens})()
        self.preprocessor_conf = {"fs": RATE, "speech_length": 30}
        self.ctc_only = ctc_only
        self._raises = raises
        self.calls = []

    def no_language(self):
        return "<eng>"

    def iter_long(self, speech, **kwargs):
        self.calls.append(kwargs)
        if self._raises:
            raise self._raises
        yield from self._utterances


def _runner_output(result, batched=False):
    return ESP2S2TInferenceRunner.format_output(result, batched=batched)


# -- the runner -------------------------------------------------------------


def test_utterances_become_text_and_segments():
    out = _runner_output([(0.0, 1.5, "hello"), (1.5, 3.0, "there")])
    assert out["text"] == "hello there"
    assert out["segments"] == [
        {"text": "hello", "start": 0.0, "end": 1.5},
        {"text": "there", "start": 1.5, "end": 3.0},
    ]


def test_a_recording_with_nothing_in_it_is_an_empty_hypothesis():
    """Scored against the reference rather than dropped, which would flatter."""
    assert _runner_output([]) == {"text": "", "segments": []}
    assert _runner_output(None) == {"text": "", "segments": []}


def test_a_batch_formats_each_recording():
    out = _runner_output([[(0.0, 1.0, "one")], []], batched=True)
    assert [o["text"] for o in out] == ["one", ""]


def test_a_malformed_utterance_is_rejected():
    with pytest.raises(TypeError, match="start, end, text"):
        _runner_output([(0.0, "only two")])
    with pytest.raises(TypeError, match="list of"):
        _runner_output("not a list")


def test_audio_is_unwrapped_for_the_backend():
    audio = Audio(np.zeros(RATE, dtype=np.float32), RATE)
    out = ESP2S2TInferenceRunner.prepare_model_inputs({"speech": audio})
    assert isinstance(out["speech"], np.ndarray)
    assert out["speech"].shape == (RATE,)


def test_a_batch_of_audio_is_unwrapped():
    audio = Audio(np.zeros(RATE, dtype=np.float32), RATE)
    out = ESP2S2TInferenceRunner.prepare_model_inputs({"speech": [audio, audio]})
    assert all(isinstance(a, np.ndarray) for a in out["speech"])


# -- the backend ------------------------------------------------------------


def test_the_language_must_be_in_the_vocabulary():
    backend = S2TLongFormBackend(model=_Model(tokens=["<eng>", "<asr>"]))
    assert backend.language_symbol("eng") == "<eng>"
    with pytest.raises(ValueError, match="ISO 639-3"):
        backend.language_symbol("xxx")


def test_a_model_that_lists_no_tokens_is_given_the_benefit_of_the_doubt():
    backend = S2TLongFormBackend(model=_Model(tokens=None))
    assert backend.language_symbol("anything") == "<anything>"


def test_an_unknown_task_names_the_targets_the_model_has():
    tokens = ["<eng>", "<asr>", "<st_deu>", "<st_jpn>"]
    backend = S2TLongFormBackend(model=_Model(tokens=tokens))
    assert backend.task_symbol("asr") == "<asr>"
    with pytest.raises(ValueError, match="deu, jpn"):
        backend.task_symbol("st_kor")


def test_no_language_falls_back_to_the_checkpoint():
    backend = S2TLongFormBackend(model=_Model(tokens=["<eng>"]))
    assert backend.language_symbol(None) == "<eng>"


def test_decode_returns_the_backend_utterances():
    model = _Model(utterances=[(0.0, 1.0, "hi")], tokens=["<eng>", "<asr>"])
    backend = S2TLongFormBackend(model=model)
    assert backend.decode(np.zeros(RATE, dtype=np.float32)) == [(0.0, 1.0, "hi")]


def test_a_clip_too_short_to_subsample_decodes_to_nothing():
    """espnet2 raises; an empty hypothesis still counts against the reference."""
    model = _Model(tokens=["<eng>"], raises=TooShortUttError("too short", 1, 2))
    backend = S2TLongFormBackend(model=model)
    assert backend.decode(np.zeros(8, dtype=np.float32)) == []


def test_previous_text_conditions_the_decode():
    model = _Model(utterances=[(0.0, 1.0, "hi")], tokens=["<eng>", "<asr>"])
    backend = S2TLongFormBackend(model=model)
    backend.decode(np.zeros(RATE, dtype=np.float32), previous_text="before")
    assert model.calls[0]["init_text"] == "before"
    assert model.calls[0]["condition_on_prev_text"] is True


def test_phones_go_window_by_window_only_on_an_encoder_decoder():
    enc_dec = S2TLongFormBackend(model=_Model(ctc_only=False))
    ctc = S2TLongFormBackend(model=_Model(ctc_only=True))
    assert enc_dec.phones_by_window("<pr>") is True
    assert ctc.phones_by_window("<pr>") is False
    assert enc_dec.phones_by_window("<asr>") is False


def test_the_batchable_window_leaves_the_last_second_out():
    backend = S2TLongFormBackend(model=_Model())
    assert backend.batchable_samples() == RATE * 29


def test_a_model_that_does_not_say_its_window_is_not_batched():
    model = _Model()
    model.preprocessor_conf = {}
    assert S2TLongFormBackend(model=model).batchable_samples() is None
