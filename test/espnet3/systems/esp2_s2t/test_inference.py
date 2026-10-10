"""The OWSM system's Inference, with a stand-in for the s2t Speech2Text."""

from __future__ import annotations

import sys
import types

# the espnet2 tests' tiny S2T model, for the one test on a real backend
from test.espnet2.bin.test_s2t_inference import (  # noqa: F401
    s2t_long_form_config_file,
    token_list,
)
from types import SimpleNamespace

import numpy as np
import pytest

from espnet3.api.inference import ModelTagError, load
from espnet3.systems.esp2_s2t.inference import Inference

VOCABULARY = ["<blank>", "<unk>", "<nolang>", "<eng>", "<jpn>", "<asr>", "<st_eng>"]


class FakeSpeech2Text:
    """Yields what a Speech2Text's iter_long would, remembering its arguments."""

    def __init__(
        self,
        utterances=((0.0, 1.5, "hello world"), (1.5, 2.0, "again")),
        tokens=VOCABULARY,
        lang_sym="<eng>",
        task_sym="<asr>",
        sample_rate=16000,
        **kwargs,
    ):
        self.utterances = list(utterances)
        self.s2t_model = SimpleNamespace(token_list=list(tokens))
        self.lang_sym = lang_sym
        self.task_sym = task_sym
        self.sample_rate = sample_rate
        self.preprocessor_conf = {}
        self.kwargs = kwargs
        self.calls = []

    def no_language(self):
        for candidate in ("<nolang>", "<unk>"):
            if candidate in self.s2t_model.token_list:
                return candidate
        raise ValueError("this checkpoint has no symbol for an unknown language")

    def iter_long(self, speech, **kwargs):
        self.calls.append((np.asarray(speech).shape, kwargs))
        yield from self.utterances

    def decode_long(self, speech, **kwargs):
        return list(self.iter_long(speech, **kwargs))


def audio(seconds=2.0, rate=16000):
    return np.zeros(int(seconds * rate), dtype=np.float32)


def test_transcribes_a_recording_into_text_and_segments():
    backend = FakeSpeech2Text()
    out = Inference(backend)(audio())
    assert out == {
        "text": "hello world again",
        "segments": [
            {"text": "hello world", "start": 0.0, "end": 1.5},
            {"text": "again", "start": 1.5, "end": 2.0},
        ],
    }
    ((shape, kwargs),) = backend.calls
    assert shape == (32000,)
    assert kwargs == {"lang_sym": "<eng>", "task_sym": None}


def test_a_language_is_passed_as_the_checkpoints_symbol():
    backend = FakeSpeech2Text()
    Inference(backend)(audio(), language="jpn")
    assert backend.calls[0][1]["lang_sym"] == "<jpn>"


def test_a_language_the_checkpoint_lacks_is_refused_before_decoding():
    backend = FakeSpeech2Text()
    with pytest.raises(ValueError, match="language 'en': this model has no <en>"):
        Inference(backend)(audio(), language="en")
    assert backend.calls == []


def test_no_language_falls_back_to_the_checkpoints_own_symbol():
    """The backend's lang_sym when the vocabulary has it; else no_language()."""
    backend = FakeSpeech2Text(lang_sym="<nolang>")
    Inference(backend)(audio())
    assert backend.calls[0][1]["lang_sym"] == "<nolang>"

    # a POWSM-like vocabulary: no <nolang>, so the checkpoint's <unk>
    backend = FakeSpeech2Text(lang_sym="<nolang>", tokens=["<unk>", "<eng>", "<asr>"])
    Inference(backend)(audio())
    assert backend.calls[0][1]["lang_sym"] == "<unk>"

    backend = FakeSpeech2Text(lang_sym="<nolang>", tokens=["<eng>", "<asr>"])
    with pytest.raises(ValueError, match="pass language=<ISO 639-3>"):
        Inference(backend)(audio())


def test_a_task_is_passed_as_the_checkpoints_symbol_and_checked():
    backend = FakeSpeech2Text()
    Inference(backend)(audio(), task="st_eng")
    assert backend.calls[0][1]["task_sym"] == "<st_eng>"
    with pytest.raises(ValueError, match="task 'st_jpn': this model has no <st_jpn>"):
        Inference(backend)(audio(), task="st_jpn")


def test_previous_text_conditions_the_model():
    backend = FakeSpeech2Text()
    Inference(backend)(audio(), previous_text="so far")
    kwargs = backend.calls[0][1]
    assert kwargs["init_text"] == "so far" and kwargs["condition_on_prev_text"] is True


def test_stream_yields_each_utterance_as_it_is_decoded():
    """One output chunk per utterance, text pieces that join into the text."""
    model = Inference(FakeSpeech2Text())
    pieces = list(model.stream([{"speech": audio()}]))
    assert [p["text"] for p in pieces] == ["hello world", " again"]
    assert [p["segments"] for p in pieces] == [
        [{"text": "hello world", "start": 0.0, "end": 1.5}],
        [{"text": "again", "start": 1.5, "end": 2.0}],
    ]
    assert "".join(p["text"] for p in pieces) == model(audio())["text"]


def test_a_recording_with_nothing_in_it_gives_empty_outputs():
    assert Inference(FakeSpeech2Text(utterances=()))(audio()) == {
        "text": "",
        "segments": [],
    }


def test_sample_rate_is_the_backends():
    assert Inference(FakeSpeech2Text(sample_rate=8000)).sample_rate == 8000


def test_a_real_tiny_model_end_to_end(
    s2t_long_form_config_file, monkeypatch  # noqa: F811
):
    """A randomly initialised S2T model, its decoder standing in for a trained one.

    The espnet2 fixture's window is one second at 2 kHz, spanned by <0.00>
    and <1.00>; the stand-in answers every window with "a" from its start
    to its end, so a 2.5 s recording is three utterances.
    """
    from espnet2.bin.s2t_inference import Speech2Text

    backend = Speech2Text(s2t_train_config=s2t_long_form_config_file, beam_size=1)
    ids = backend.converter.token2id
    tokens = [ids["<eng>"], ids["<asr>"], ids["<0.00>"], ids["a"], ids["<1.00>"]]
    monkeypatch.setattr(
        backend, "__call__", lambda **kw: [("a", ["a"], list(tokens), "a", None)]
    )
    model = Inference(backend)
    assert model.sample_rate == 2000
    out = model(np.zeros(5000, dtype=np.float32), language="eng")
    assert out["text"] == "a a a"
    assert [(s["start"], s["end"]) for s in out["segments"]] == [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
    ]


class PublishedSpeech2Text(FakeSpeech2Text):
    """A backend class that reads its own toolkit's published checkpoints."""

    loaded = []

    @classmethod
    def from_pretrained(cls, model_tag=None, device="cpu", **kwargs):
        from espnet2.utils.pretrained import download_pretrained

        cls.loaded.append((model_tag, device, dict(kwargs)))
        kwargs.update(download_pretrained(model_tag))  # as Speech2Text does
        if "asr_train_config" in kwargs:  # what a tag for another model brings
            raise TypeError(
                "__init__() got an unexpected keyword argument 'asr_train_config'"
            )
        return cls(**kwargs)


def _espnet2_downloader(monkeypatch, artifacts):
    module = types.ModuleType("espnet_model_zoo.downloader")

    class ModelDownloader:
        def download_and_unpack(self, tag):
            return dict(artifacts)

    module.ModelDownloader = ModelDownloader
    monkeypatch.setitem(sys.modules, "espnet_model_zoo.downloader", module)


def test_load_reads_a_published_owsm_checkpoint_when_the_caller_names_s2t(
    monkeypatch,
):
    import espnet2.bin.s2t_inference as s2t_inference

    monkeypatch.setattr(s2t_inference, "Speech2Text", PublishedSpeech2Text)
    _espnet2_downloader(monkeypatch, {"s2t_train_config": "c.yaml"})
    PublishedSpeech2Text.loaded.clear()
    model = load("espnet/owsm_like", system="s2t", beam_size=5)
    assert isinstance(model, Inference)
    assert PublishedSpeech2Text.loaded == [
        ("espnet/owsm_like", "cpu", {"beam_size": 5})
    ]
    assert model(audio())["text"] == "hello world again"


def test_an_asr_checkpoint_given_to_the_s2t_system_is_a_model_tag_error(monkeypatch):
    import espnet2.bin.s2t_inference as s2t_inference

    monkeypatch.setattr(s2t_inference, "Speech2Text", PublishedSpeech2Text)
    _espnet2_downloader(monkeypatch, {"asr_train_config": "c.yaml"})
    with pytest.raises(ModelTagError, match="does not look like a model for"):
        load("espnet/an_asr_model", system="s2t")
