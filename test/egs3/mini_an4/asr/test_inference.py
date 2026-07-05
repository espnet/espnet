import importlib
import sys
import types


def _load_inference_module(monkeypatch):
    fake_module = types.ModuleType("espnet2.bin.asr_transducer_inference")

    class FakeSpeech2Text:
        def __call__(self, speech):
            return [speech]

        def hypotheses_to_results(self, hypotheses):
            return hypotheses

    fake_module.Speech2Text = FakeSpeech2Text
    monkeypatch.setitem(
        sys.modules,
        "espnet2.bin.asr_transducer_inference",
        fake_module,
    )
    sys.modules.pop("egs3.mini_an4.asr.src.inference", None)
    return importlib.import_module("egs3.mini_an4.asr.src.inference")


def test_transducer_inference_wrapper_converts_hypotheses(monkeypatch):
    inference_mod = _load_inference_module(monkeypatch)
    calls = {}

    def fake_base_call(self, speech):
        calls["speech"] = speech
        return ["raw-hypothesis"]

    def fake_hypotheses_to_results(self, hypotheses):
        calls["hypotheses"] = hypotheses
        return [("wrapped text", ["wrapped", "text"], [1, 2], object())]

    monkeypatch.setattr(
        inference_mod.TransducerSpeech2Text,
        "__call__",
        fake_base_call,
    )
    monkeypatch.setattr(
        inference_mod.TransducerSpeech2Text,
        "hypotheses_to_results",
        fake_hypotheses_to_results,
    )

    wrapper = object.__new__(inference_mod.TransducerInferenceWrapper)
    result = wrapper("audio")

    assert calls["speech"] == "audio"
    assert calls["hypotheses"] == ["raw-hypothesis"]
    assert result[0][0] == "wrapped text"


def test_build_output_transducer_uses_best_text(monkeypatch):
    inference_mod = _load_inference_module(monkeypatch)
    output = inference_mod.build_output_transducer(
        {"utt_id": "utt1", "text": "ref"},
        [("best text", ["best", "text"], [1, 2], object())],
        0,
    )

    assert output == {"utt_id": "utt1", "hyp": "best text", "ref": "ref"}


def test_build_output_transducer_falls_back_to_tokens(monkeypatch):
    inference_mod = _load_inference_module(monkeypatch)
    output = inference_mod.build_output_transducer(
        {"utt_id": "utt1", "text": "ref"},
        [(None, ["a", "b"], [1, 2], object())],
        0,
    )

    assert output == {"utt_id": "utt1", "hyp": "a b", "ref": "ref"}
