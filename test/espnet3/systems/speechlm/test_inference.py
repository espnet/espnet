"""Exercise Bagpiper's ESPnet3 contracts with a small native decoder stand-in."""

from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch

from espnet2.bin.speechlm_inference import PAPER_DECODING, LocalBagpiper
from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner
from espnet3.systems.speechlm.inference import BagpiperInference


@pytest.fixture
def native(monkeypatch):
    preprocessor = Mock()
    preprocessor.collate_fn.return_value = {"keys": [], "seqs": torch.tensor([0])}
    model = Mock()
    model.inference.return_value = ([["assistant", "text", ["first\nsecond"]]], None)
    native = LocalBagpiper(
        model,
        preprocessor,
        PAPER_DECODING,
        PAPER_DECODING,
        "cpu",
        torch.float32,
        tts_system="Generation instructions",
    )
    monkeypatch.setattr(LocalBagpiper, "from_pretrained", Mock(return_value=native))
    return native


@pytest.mark.parametrize("channels", [1, 2])
def test_understand_through_runner(native, channels):
    backend = BagpiperInference("understand", decoding={"max_steps": 16})
    waveform = np.arange(channels * 32, dtype=np.float32).reshape(channels, 32)
    request = {
        "utt_id": "question",
        "speech": Audio(waveform, 8000),
        "prompt": "What do you hear?",
        "system": "Answer the question",
    }
    output = InferenceRunner.forward(
        0,
        dataset=[request],
        model=backend,
        input_key=["prompt", "speech", "system"],
        inference_contract={
            "inputs": {
                "prompt": {"kind": "text"},
                "speech": {"kind": "audio"},
                "system": {"kind": "text", "optional": True},
            },
            "outputs": {"text": {"kind": "text"}},
        },
    )
    assert output == {"utt_id": "question", "text": "first second"}
    [(key, sample)] = native.preprocessor.collate_fn.call_args.args[0]
    assert key[0] == "dialogue"
    system, audio, prompt = sample["dialogue"]
    assert system == ("system", "text", "Answer the question")
    assert audio[:2] == ("user", "audio")
    samples, rate = audio[2]
    np.testing.assert_array_equal(samples, waveform)
    assert rate == 8000
    assert prompt == ("user", "text", "What do you hear?")
    config = native.model.inference.call_args.args[0]
    assert config["enforce_modality"] == ["text"]
    assert config["text"]["max_step"] == 16


def test_generate_through_runner(native, tmp_path):
    samples = torch.tensor([[[0.1234567, -0.2345678, 0.5]]])
    native.model.inference.return_value = (
        [
            ["assistant", "text", ["A bell"]],
            ["assistant", "audio", (samples, torch.tensor([2]), 24000)],
        ],
        None,
    )
    output = InferenceRunner.forward(
        0,
        dataset=[{"utt_id": "sound", "prompt": "A distant bell."}],
        model=BagpiperInference("generate"),
        input_key=["prompt", "system"],
        inference_contract={
            "inputs": {
                "prompt": {"kind": "text"},
                "system": {"kind": "text", "optional": True},
            },
            "outputs": {"text": {"kind": "text"}, "audio": {"kind": "audio"}},
        },
    )
    assert output["utt_id"] == "sound"
    assert output["text"] == "A bell"
    assert output["audio"].rate == 24000
    np.testing.assert_array_equal(output["audio"].array, samples[0, 0, :2].numpy())
    [(_, sample)] = native.preprocessor.collate_fn.call_args.args[0]
    assert sample["dialogue"] == [("user", "text", "A distant bell.")]
    assert native.model.inference.call_args.args[0]["enforce_modality"] == [
        "text",
        "audio",
    ]
    writers = InferenceRunner.open_writers(tmp_path)
    try:
        InferenceRunner.write_record(writers, output, {}, hyp_key=["text", "audio"])
    finally:
        InferenceRunner.close_writers(writers, {})
    assert (tmp_path / "text.scp").read_text() == "sound A bell\n"
    key, filename = (tmp_path / "audio.scp").read_text().strip().split(maxsplit=1)
    assert key == "sound"
    waveform, rate = sf.read(filename)
    assert rate == 24000
    np.testing.assert_allclose(waveform, samples[0, 0, :2], atol=1 / 32768)


def test_rejects_invalid_mode_before_loading(native):
    with pytest.raises(ValueError, match="mode must be"):
        BagpiperInference("encode")
    LocalBagpiper.from_pretrained.assert_not_called()


def test_missing_input_or_generated_audio(native):
    with pytest.raises(ValueError, match="requires input speech"):
        BagpiperInference("understand")("What is that?")
    with pytest.raises(RuntimeError, match="without emitting audio"):
        BagpiperInference("generate")("An empty room.", system="")
    [(_, sample)] = native.preprocessor.collate_fn.call_args.args[0]
    assert sample["dialogue"] == [
        ("system", "text", ""),
        ("user", "text", "An empty room."),
    ]


def test_empty_generated_waveform_fails_before_writing(native):
    native.model.inference.return_value = (
        [["assistant", "audio", (torch.zeros(1, 1, 4), torch.tensor([0]), 24000)]],
        None,
    )
    with pytest.raises(RuntimeError, match="without emitting audio"):
        BagpiperInference("generate")("A bell.")


def test_partial_override_preserves_native_yaml(native):
    native.text_config = {
        **PAPER_DECODING,
        "text": {**PAPER_DECODING["text"], "temperature": 0.35, "topk": 7},
        "audio": {**PAPER_DECODING["audio"], "temperature": 0.25, "topk": 9},
    }
    model = BagpiperInference("understand", decoding={"max_steps": 4})
    model("What's that sound?", speech=Audio(np.zeros(32), 16000))
    used = native.model.inference.call_args.args[0]
    for modality in ("text", "audio"):
        assert used[modality] == {**native.text_config[modality], "max_step": 4}
    assert native.text_config["text"]["max_step"] == 2048


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"max_steps": 0}, "max_steps"),
        ({"max_tokens": 4}, "Use max_steps"),
        ({"topk": 0}, "topk"),
        ({"audio_temperature": float("nan")}, "audio_temperature"),
    ],
)
def test_bad_decoding_fails_before_loading(native, overrides, message):
    with pytest.raises(ValueError, match=message):
        BagpiperInference("generate", decoding=overrides)
    LocalBagpiper.from_pretrained.assert_not_called()
