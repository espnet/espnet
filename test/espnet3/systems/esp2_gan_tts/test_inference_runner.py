"""The GAN-TTS runner adapts Text2Speech's outputs to the contract."""

import numpy as np
import pytest
import torch

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.esp2_gan_tts.src.inference_runner import TTSInferenceRunner

CONTRACT = {
    "inputs": {"text": {"kind": "text"}, "spembs": {"kind": "vector"}},
    "outputs": {"wav": {"kind": "audio", "sample_rate": 22050, "channels": 1}},
}


class FakeText2Speech:
    """Return Text2Speech's output shape without building a model."""

    def __init__(self):
        """Keep what the runner handed over."""
        self.seen = None

    def __call__(self, text, spembs=None):
        """Echo a waveform whose length follows the text, beside diagnostics."""
        self.seen = {"text": text, "spembs": spembs}
        return {
            "wav": torch.linspace(-0.5, 0.5, 100 * len(text)),
            "att_w": None,
            "feat_gen": torch.zeros(3, 80),
            "duration": torch.ones(len(text)),
        }


def test_prepare_model_inputs_passes_the_checked_values_through():
    """Text and the embedding reach the backend unchanged."""
    spembs = np.ones(192, dtype=np.float32)
    out = TTSInferenceRunner.prepare_model_inputs({"text": "hi", "spembs": spembs})
    assert out["text"] == "hi"
    assert out["spembs"] is spembs


def test_prepare_model_inputs_refuses_a_batch():
    """A list of texts is refused with the batch_size hint."""
    with pytest.raises(ValueError, match="batch_size: null"):
        TTSInferenceRunner.prepare_model_inputs({"text": ["a", "b"]})


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_prepare_model_inputs_refuses_empty_text(text):
    """Empty or whitespace text is refused before Text2Speech sees it."""
    with pytest.raises(ValueError, match="'text' is empty"):
        TTSInferenceRunner.prepare_model_inputs({"text": text, "spembs": np.ones(2)})


def test_format_output_keeps_only_the_waveform_as_float32():
    """Only `wav` survives, as a 1-D float32 array."""
    out = TTSInferenceRunner.format_output(
        {"wav": torch.ones(240), "att_w": None, "duration": torch.ones(3)},
        batched=False,
    )
    assert list(out) == ["wav"]
    assert out["wav"].dtype == np.float32
    assert out["wav"].shape == (240,)


@pytest.mark.parametrize("result", [None, "wav", ["wav"]])
def test_format_output_rejects_a_non_mapping(result):
    """Anything but Text2Speech's dict is a type error."""
    with pytest.raises(TypeError, match="must return a dict"):
        TTSInferenceRunner.format_output(result, batched=False)


def test_format_output_requires_wav():
    """A result without `wav` is a key error."""
    with pytest.raises(KeyError, match="wav"):
        TTSInferenceRunner.format_output({"att_w": None}, batched=False)
    with pytest.raises(KeyError, match="wav"):
        TTSInferenceRunner.format_output({"wav": None}, batched=False)


def test_format_output_refuses_a_batch():
    """A batched result is refused with the batch_size hint."""
    with pytest.raises(ValueError, match="batch_size: null"):
        TTSInferenceRunner.format_output([{"wav": 1}], batched=True)


def test_forward_runs_the_contract_around_the_backend():
    """The shared forward checks the inputs, calls the backend and checks the output."""
    model = FakeText2Speech()
    dataset = [
        {
            "utt_id": "u1",
            "text": "hello",
            "spembs": torch.ones(192),
            "wav_path": "/data/u1.wav",
        }
    ]

    result = TTSInferenceRunner.forward(
        0,
        dataset=dataset,
        model=model,
        input_key=["text", "spembs"],
        inference_contract=CONTRACT,
    )

    # The raw text went through; the vector kind handed over a NumPy array.
    assert model.seen["text"] == "hello"
    assert isinstance(model.seen["spembs"], np.ndarray)
    assert model.seen["spembs"].shape == (192,)
    # The declared rate turned the bare waveform into an Audio; the
    # diagnostics Text2Speech returned beside it were dropped.
    assert set(result) == {"wav", "utt_id"}
    assert isinstance(result["wav"], Audio)
    assert result["wav"].rate == 22050
    assert result["wav"].channels == 1
    assert result["wav"].array.shape == (500,)
    assert result["utt_id"] == "u1"


def test_forward_refuses_a_bad_embedding():
    """A 2-D embedding is refused by the contract before the backend runs."""
    model = FakeText2Speech()
    dataset = [{"utt_id": "u1", "text": "hello", "spembs": np.ones((2, 192))}]

    with pytest.raises(TypeError, match="spembs"):
        TTSInferenceRunner.forward(
            0,
            dataset=dataset,
            model=model,
            input_key=["text", "spembs"],
            inference_contract=CONTRACT,
        )
    assert model.seen is None
