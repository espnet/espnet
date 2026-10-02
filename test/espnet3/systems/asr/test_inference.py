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

    def __init__(self, fs="16k", per_speaker=False, **kwargs):
        self.asr_train_args = SimpleNamespace(frontend_conf={"fs": fs})
        self.per_speaker = per_speaker
        self.kwargs = kwargs
        self.seen = None

    def __call__(self, speech):
        self.seen = speech
        if isinstance(speech, list):
            return self.batch_decode(speech)
        best = ("hello world", ["hello", "world"], [1, 2], None)
        nbest = [best, ("hello", ["hello"], [1], None)]
        return [nbest, nbest] if self.per_speaker else nbest

    def batch_decode(self, speeches):
        self.batches = getattr(self, "batches", []) + [len(speeches)]
        return [[(f"utt{i}", None, None, None)] for i in range(len(speeches))]


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
    import espnet3.systems.base.backend_inference as base

    backend = FakeSpeech2Text()
    calls = []

    def locate_pack(tag_or_dir):
        calls.append(("locate", str(tag_or_dir)))
        return tmp_path

    def load_backend(pack_dir, *, device=None):
        calls.append(("build", str(pack_dir), device))
        return backend

    monkeypatch.setattr(base, "locate_pack", locate_pack)
    monkeypatch.setattr(base, "load_backend", load_backend)

    assert Inference.from_pretrained(tmp_path, device="cpu").backend is backend
    assert Inference.from_pretrained("org/model", device="cuda:0").backend is backend
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

    monkeypatch.setattr(asr_inference, "Speech2Text", FakeSpeech2Text)
    model = Inference(asr_train_config="c.yaml", asr_model_file="m.pth", beam_size=3)
    assert model.backend.kwargs == {
        "device": "cpu",
        "asr_train_config": "c.yaml",
        "asr_model_file": "m.pth",
        "beam_size": 3,
    }
    assert model(np.zeros(16))["text"] == "hello world"
    with pytest.raises(TypeError, match="but a built one was too"):
        Inference(FakeSpeech2Text(), beam_size=3)


def test_builds_the_transducer_speech2text_when_told(monkeypatch):
    import espnet2.bin.asr_transducer_inference as transducer

    monkeypatch.setattr(transducer, "Speech2Text", FakeSpeech2Text)
    model = Inference(
        backend_class="espnet2.bin.asr_transducer_inference.Speech2Text",
        asr_train_config="c.yaml",
        return_decoded_hyp=True,
    )
    assert model.backend.kwargs == {
        "device": "cpu",
        "asr_train_config": "c.yaml",
        "return_decoded_hyp": True,
    }
    assert model(np.zeros(16))["text"] == "hello world"


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
    assert batched == [{"text": "utt0"}, {"text": "utt1"}]  # one batch_decode


def test_the_infer_stage_provider_builds_it_from_inference_yaml(monkeypatch):
    import espnet2.bin.asr_inference as asr_inference

    monkeypatch.setattr(asr_inference, "Speech2Text", FakeSpeech2Text)
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
    assert model.backend.kwargs["asr_train_config"] == "exp/config.yaml"


def test_a_batch_is_one_beam_search_when_the_backend_can():
    backend = FakeSpeech2Text()
    model = Inference(backend)
    items = [{"speech": np.zeros(16, dtype=np.float32)}] * 3
    assert [o["text"] for o in model.batch(items)] == ["utt0", "utt1", "utt2"]
    assert backend.batches == [3]
    assert model.batch(items[:1]) == [{"text": "hello world"}]  # one item: run

    class NoBatch(FakeSpeech2Text):
        batch_decode = None  # the transducer Speech2Text has none

    assert [o["text"] for o in Inference(NoBatch()).batch(items)] == ["hello world"] * 3


def test_from_pretrained_returns_the_bundles_own_inference(tmp_path, monkeypatch):
    import espnet3.systems.base.backend_inference as base

    monkeypatch.setattr(base, "locate_pack", lambda t: tmp_path)
    ready = Inference(FakeSpeech2Text())
    monkeypatch.setattr(base, "load_backend", lambda p, *, device=None: ready)
    assert Inference.from_pretrained(tmp_path) is ready

    from espnet3.api.inference import Field
    from espnet3.systems.base.backend_inference import BackendInference

    class Other(BackendInference):
        inputs = (Field("text", "text"),)
        outputs = (Field("speech", "audio"),)

        def run(self, text):
            return {}

    monkeypatch.setattr(base, "load_backend", lambda p, *, device=None: Other(object()))
    with pytest.raises(TypeError, match="builds Other, not Inference"):
        Inference.from_pretrained(tmp_path)
