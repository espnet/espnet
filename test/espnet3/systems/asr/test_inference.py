"""The ASR system's Inference, with a stand-in for Speech2Text."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from espnet3.api.inference import Audio
from espnet3.systems.asr.inference import Inference
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import InferenceRunner


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
    import espnet3.systems.asr.inference as module

    backend = FakeSpeech2Text()
    calls = []

    def locate_pack(tag_or_dir):
        calls.append(("locate", str(tag_or_dir)))
        return tmp_path

    def load_backend(pack_dir, *, device=None):
        calls.append(("build", str(pack_dir), device))
        return backend

    monkeypatch.setattr(module, "locate_pack", locate_pack)
    monkeypatch.setattr(module, "load_backend", load_backend)

    assert Inference.from_pretrained(tmp_path, device="cpu").speech2text is backend
    assert (
        Inference.from_pretrained("org/model", device="cuda:0").speech2text is backend
    )
    assert calls == [
        ("locate", str(tmp_path)),
        ("build", str(tmp_path), "cpu"),
        ("locate", "org/model"),
        ("build", str(tmp_path), "cuda:0"),
    ]
    with pytest.raises(TypeError, match="unexpected arguments"):
        Inference.from_pretrained(tmp_path, beam_size=3)


def test_builds_a_speech2text_from_its_own_arguments(monkeypatch):
    import espnet2.bin.asr_inference as asr_inference

    built = {}

    def fake(**kwargs):
        built.update(kwargs)
        return FakeSpeech2Text()

    monkeypatch.setattr(asr_inference, "Speech2Text", fake)
    model = Inference(asr_train_config="c.yaml", asr_model_file="m.pth", beam_size=3)
    assert built == {
        "device": "cpu",
        "asr_train_config": "c.yaml",
        "asr_model_file": "m.pth",
        "beam_size": 3,
    }
    assert model(np.zeros(16))["text"] == "hello world"
    with pytest.raises(TypeError, match="given, but a speech2text was too"):
        Inference(FakeSpeech2Text(), beam_size=3)


def test_the_infer_stage_runner_calls_it_as_it_is():
    """InferenceRunner needs no output_fn: the contract's mapping is written."""
    model = Inference(FakeSpeech2Text())
    dataset = {
        i: {"speech": np.zeros(160, dtype=np.float32), "text": "ref"} for i in range(3)
    }
    assert InferenceRunner.forward(
        0, dataset=dataset, model=model, input_key="speech"
    ) == {"text": "hello world"}
    batched = InferenceRunner.forward(
        [1, 2], dataset=dataset, model=model, input_key="speech"
    )
    assert batched == [{"text": "hello world"}, {"text": "hello world"}]


def test_the_infer_stage_provider_builds_it_from_inference_yaml(monkeypatch):
    import espnet2.bin.asr_inference as asr_inference

    built = {}

    def fake(**kwargs):
        built.update(kwargs)
        return FakeSpeech2Text()

    monkeypatch.setattr(asr_inference, "Speech2Text", fake)
    config = OmegaConf.create(
        {
            "device": "cpu",
            "model": {
                "_target_": "espnet3.systems.asr.inference.Inference",
                "asr_train_config": "exp/config.yaml",
                "asr_model_file": "exp/model.pth",
            },
        }
    )
    model = InferenceProvider.build_model(config)
    assert isinstance(model, Inference)
    assert built["asr_train_config"] == "exp/config.yaml" and built["device"] == "cpu"
