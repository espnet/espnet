"""The inference contract, checked without a model.

A fake transcriber stands in for a system, so what is tested is what the
base class promises every front end: binding, conversion and checking.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

import espnet3.api.inference as inference_api
from espnet3.api.inference import (
    Audio,
    Field,
    InferenceAPI,
    check_contract,
    load,
    locate_pack,
)


class Echo(InferenceAPI):
    """A transcriber that reports what it was given."""

    inputs = (Field("speech", "audio"), Field("prompt", "text", optional=True))
    outputs = (Field("text", "text"),)

    def __init__(self, rate: int = 16000) -> None:
        self.rate = rate
        self.seen = None

    @classmethod
    def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
        return cls()

    @property
    def sample_rate(self) -> int:
        return self.rate

    def run(self, speech: Audio, prompt: str = "") -> dict:
        self.seen = speech
        return {"text": f"{speech.seconds:.1f}s@{speech.rate}{prompt}", "n": 1}


# --- Field and Audio -------------------------------------------------------


def test_field_rejects_unknown_kind_and_bad_name():
    with pytest.raises(ValueError, match="kind"):
        Field("x", "image")
    with pytest.raises(ValueError, match="identifier"):
        Field("not a name", "text")


def test_field_label_defaults_to_spaced_name():
    assert Field("reference_speech", "audio").label == "Reference speech"
    assert Field("speech", "audio", "Mic").label == "Mic"


def test_audio_normalises_pcm_and_channels():
    pcm = np.full((8, 2), 16384, dtype=np.int16)
    audio = Audio(pcm, 8000)
    assert audio.array.dtype == np.float32
    assert audio.array.shape == (8,)
    assert audio.array[0] == pytest.approx(16384 / 32767)
    assert audio.seconds == pytest.approx(0.001)


def test_audio_takes_channels_on_either_axis():
    assert len(Audio(np.zeros((100, 2)), 8000).array) == 100  # soundfile, Gradio
    assert len(Audio(np.zeros((1, 100)), 8000).array) == 100  # torchaudio


def test_audio_rejects_bad_shapes_and_rates():
    with pytest.raises(ValueError, match="1-D"):
        Audio(np.zeros((2, 2, 2)), 16000)
    with pytest.raises(ValueError, match="positive"):
        Audio(np.zeros(4), 0)


def test_audio_to_resamples_only_when_needed():
    audio = Audio(np.zeros(16000, dtype=np.float32), 16000)
    assert audio.to(16000) is audio
    assert len(audio.to(8000).array) == 8000


def test_audio_coerce_every_shape_a_caller_holds(tmp_path):
    wav = tmp_path / "a.wav"
    soundfile.write(wav, np.zeros((8000, 2), dtype=np.float32), 8000)
    samples = np.zeros(8000, dtype=np.float32)

    from_path = Audio.coerce(str(wav), 16000)
    from_gradio = Audio.coerce((8000, (samples * 32767).astype(np.int16)), 16000)
    from_tensor = Audio.coerce(torch.zeros(16000), 16000)
    from_array = Audio.coerce(samples, 8000)

    for audio in (from_path, from_gradio, from_tensor):
        assert audio.rate == 16000 and len(audio.array) == 16000
    assert from_array.rate == 8000 and len(from_array.array) == 8000
    with pytest.raises(TypeError, match="audio must be"):
        Audio.coerce(42, 16000)


# --- the contract ----------------------------------------------------------


def test_contract_is_checked_when_the_class_is_defined():
    with pytest.raises(TypeError, match="tuple of Field"):

        class NotFields(InferenceAPI):
            inputs = ["speech"]
            outputs = (Field("text", "text"),)

    with pytest.raises(TypeError, match="at least one field"):

        class Nothing(InferenceAPI):
            inputs = ()
            outputs = (Field("text", "text"),)

    with pytest.raises(TypeError, match="required fields before optional"):

        class OptionalFirst(InferenceAPI):
            inputs = (Field("prompt", "text", optional=True), Field("speech", "audio"))
            outputs = (Field("text", "text"),)

    with pytest.raises(TypeError, match="repeats"):

        class Twice(InferenceAPI):
            inputs = (Field("speech", "audio"), Field("speech", "audio", optional=True))
            outputs = (Field("text", "text"),)

    with pytest.raises(TypeError, match="optional"):

        class OptionalOut(InferenceAPI):
            inputs = (Field("speech", "audio"),)
            outputs = (Field("text", "text", optional=True),)


def test_an_intermediate_base_that_declares_nothing_is_allowed():
    class Base(InferenceAPI):
        pass

    with pytest.raises(TypeError):
        check_contract(Base)


def test_a_conversation_model_declares_no_task():
    class Chat(InferenceAPI):
        inputs = (Field("messages", "text"),)
        outputs = (Field("messages", "text"),)

        @classmethod
        def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
            return cls()

        sample_rate = 16000

        def run(self, messages):
            return {"messages": messages + " and reply"}

    assert Chat()("hi")["messages"] == "hi and reply"


# --- calling ---------------------------------------------------------------


def test_call_binds_by_position_or_name_and_resamples():
    model = Echo(rate=8000)
    samples = np.zeros(16000, dtype=np.float32)
    assert model((16000, samples))["text"] == "1.0s@8000"
    assert model(speech=(16000, samples), prompt="!")["text"] == "1.0s@8000!"
    assert model.seen.rate == 8000
    assert model((16000, samples))["n"] == 1  # extra outputs pass through


def test_call_rejects_what_the_contract_does_not_allow():
    model = Echo()
    with pytest.raises(TypeError, match="needs 'speech'"):
        model()
    with pytest.raises(TypeError, match="no input \\['lang'\\]"):
        model(np.zeros(8), lang="en")
    with pytest.raises(TypeError, match="both by position and by name"):
        model(np.zeros(8), speech=np.zeros(8))
    with pytest.raises(TypeError, match="at most 2"):
        model(np.zeros(8), "", "extra")
    with pytest.raises(TypeError, match="must be str"):
        model(np.zeros(8), prompt=3)


def test_call_checks_what_run_returns():
    class Wrong(Echo):
        def run(self, speech, prompt=""):
            return ["not", "a", "mapping"]

    class Missing(Echo):
        def run(self, speech, prompt=""):
            return {"txt": "typo"}

    class NotText(Echo):
        def run(self, speech, prompt=""):
            return {"text": 7}

    for cls, err in ((Wrong, TypeError), (Missing, RuntimeError), (NotText, TypeError)):
        with pytest.raises(err):
            cls()(np.zeros(8))


def test_audio_output_is_wrapped_at_the_model_rate():
    class Enhancer(InferenceAPI):
        inputs = (Field("speech", "audio"),)
        outputs = (Field("speech", "audio"),)

        @classmethod
        def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
            return cls()

        sample_rate = 8000

        def run(self, speech):
            return {"speech": speech.array * 0.5}

    out = Enhancer()(np.ones(8, dtype=np.float32))["speech"]
    assert isinstance(out, Audio) and out.rate == 8000
    assert out.array[0] == pytest.approx(0.5)


def test_segments_output_is_checked():
    class Aligner(InferenceAPI):
        inputs = (Field("speech", "audio"), Field("text", "text"))
        outputs = (Field("segments", "segments"),)

        @classmethod
        def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
            return cls()

        sample_rate = 16000

        def __init__(self, good=True):
            self.good = good

        def run(self, speech, text):
            if self.good:
                return {"segments": [{"text": text, "start": 0.0, "end": 1.0}]}
            return {"segments": [(0.0, 1.0)]}

    assert Aligner()(np.zeros(16), "hi")["segments"][0]["text"] == "hi"
    with pytest.raises(TypeError, match="list of dicts"):
        Aligner(good=False)(np.zeros(16), "hi")


# --- load ------------------------------------------------------------------


def _pack(tmp_path: Path, system: str | None) -> Path:
    pack = tmp_path / "pack"
    (pack / "conf").mkdir(parents=True, exist_ok=True)
    meta = "yaml_files:\n  inference_config: conf/inference.yaml\n"
    if system:
        meta += f"system: {system}\n"
    (pack / "meta.yaml").write_text(meta)
    (pack / "conf" / "inference.yaml").write_text("recipe_dir: .\n")
    return pack


def _install_fake_system(monkeypatch, name: str, cls: type | None) -> None:
    module = types.ModuleType(f"espnet3.systems.{name}.inference")
    if cls is not None:
        module.Inference = cls
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_load_finds_the_system_named_in_meta(tmp_path, monkeypatch):
    _install_fake_system(monkeypatch, "echo", Echo)
    model = load(_pack(tmp_path, "echo"))
    assert isinstance(model, Echo)


def test_load_needs_a_system_name_from_somewhere(tmp_path, monkeypatch):
    _install_fake_system(monkeypatch, "echo", Echo)
    with pytest.raises(ValueError, match="does not name its system"):
        load(_pack(tmp_path, None))
    assert isinstance(load(_pack(tmp_path, None), system="echo"), Echo)


def test_load_follows_a_renamed_system(tmp_path, monkeypatch):
    _install_fake_system(monkeypatch, "esp2_echo", Echo)
    monkeypatch.setitem(inference_api.SYSTEM_ALIASES, "echo", "esp2_echo")
    assert isinstance(load(_pack(tmp_path, "echo")), Echo)
    assert isinstance(load(_pack(tmp_path, None), system="echo"), Echo)


def test_load_rejects_a_system_without_the_class(tmp_path, monkeypatch):
    _install_fake_system(monkeypatch, "blank", None)
    _install_fake_system(monkeypatch, "wrong", types.SimpleNamespace)
    for name in ("blank", "wrong"):
        with pytest.raises(ImportError, match="defines no Inference"):
            load(_pack(tmp_path, name))


def test_load_says_when_the_system_has_no_inference_module(tmp_path, monkeypatch):
    with pytest.raises(ImportError, match="system 'nosuch' has no Inference yet"):
        load(_pack(tmp_path, "nosuch"))

    import importlib

    def import_module(name):
        raise ModuleNotFoundError("No module named 'somedep'", name="somedep")

    monkeypatch.setattr(importlib, "import_module", import_module)
    with pytest.raises(ModuleNotFoundError, match="somedep"):
        load(_pack(tmp_path, "echo"))


def test_locate_pack_downloads_a_tag(tmp_path, monkeypatch):
    pack = _pack(tmp_path, "echo")

    class Downloader:
        def download_and_unpack(self, tag):
            assert tag == "org/model"
            return {"inference_config": str(pack / "conf" / "inference.yaml")}

    import espnet_model_zoo.downloader as downloader

    monkeypatch.setattr(downloader, "ModelDownloader", Downloader)
    assert locate_pack("org/model") == pack.resolve()
    assert locate_pack(pack) == pack.resolve()

    class Empty:
        def download_and_unpack(self, tag):
            return {}

    monkeypatch.setattr(downloader, "ModelDownloader", Empty)
    with pytest.raises(RuntimeError, match="not a pack_model bundle"):
        locate_pack("org/other")
