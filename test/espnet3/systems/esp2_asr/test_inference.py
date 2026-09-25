"""The ASR system's Inference, with a stand-in for Speech2Text."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from espnet3.api.inference import Audio
from espnet3.publication.inference_model import InferenceModel
from espnet3.systems.asr.inference import Inference


class FakeSpeech2Text:
    """Returns the n-best list Speech2Text does, remembering its input."""

    def __init__(self, fs="16k", per_speaker=False):
        self.asr_train_args = SimpleNamespace(frontend_conf={"fs": fs})
        self.per_speaker = per_speaker
        self.seen = None

    def __call__(self, speech):
        self.seen = speech
        best = ("hello world", ["hello", "world"], [1, 2], None)
        nbest = [best, ("hello", ["hello"], [1], None)]
        return [nbest, nbest] if self.per_speaker else nbest


def test_transcribes_the_best_hypothesis():
    backend = FakeSpeech2Text()
    model = Inference(backend)
    out = model(np.zeros(16000, dtype=np.float32))
    assert out == {"text": "hello world"}
    assert isinstance(backend.seen, np.ndarray) and backend.seen.dtype == np.float32


def test_first_speaker_of_a_joint_model():
    assert Inference(FakeSpeech2Text(per_speaker=True))(np.zeros(16))["text"] == (
        "hello world"
    )


def test_sample_rate_comes_from_the_frontend_config():
    assert Inference(FakeSpeech2Text(fs="16k")).sample_rate == 16000
    assert Inference(FakeSpeech2Text(fs=8000)).sample_rate == 8000
    assert Inference(SimpleNamespace()).sample_rate == 16000


def test_audio_is_resampled_to_the_frontend_rate():
    backend = FakeSpeech2Text(fs=8000)
    Inference(backend)(Audio(np.zeros(16000, dtype=np.float32), 16000))
    assert len(backend.seen) == 8000


def test_from_pretrained_takes_a_directory_or_a_tag(tmp_path, monkeypatch):
    backend = FakeSpeech2Text()
    calls = []

    def from_packed(pack_dir, device=None, **kw):
        calls.append(("packed", str(pack_dir), device))
        return SimpleNamespace(model=backend)

    def from_pretrained(tag, device=None, **kw):
        calls.append(("tag", tag, device))
        return SimpleNamespace(model=backend)

    monkeypatch.setattr(InferenceModel, "from_packed", staticmethod(from_packed))
    monkeypatch.setattr(
        InferenceModel, "from_pretrained", staticmethod(from_pretrained)
    )

    assert Inference.from_pretrained(tmp_path, device="cpu").speech2text is backend
    assert (
        Inference.from_pretrained("org/model", device="cuda:0").speech2text is backend
    )
    assert calls == [("packed", str(tmp_path), "cpu"), ("tag", "org/model", "cuda:0")]
    with pytest.raises(TypeError, match="unexpected arguments"):
        Inference.from_pretrained(tmp_path, beam_size=3)
