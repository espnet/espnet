"""The codec runner adapts AudioCoding's inputs and outputs to the contract."""

import numpy as np
import pytest
import torch

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.esp2_codec.src.inference_runner import CodecInferenceRunner

CONTRACT = {
    "inputs": {"audio": {"kind": "audio", "sample_rate": 24000, "channels": 1}},
    "outputs": {"wav": {"kind": "audio", "sample_rate": 24000, "channels": 1}},
}


class FakeAudioCoding:
    """Return AudioCoding's output shape without building a codec."""

    def __init__(self):
        """Keep what the runner handed over."""
        self.seen = None

    def __call__(self, audio):
        """Echo the waveform back as ``resyn_audio`` beside fake codes."""
        self.seen = audio
        return {
            "codes": torch.zeros(1, 2, 3),
            "resyn_audio": torch.as_tensor(audio) * 0.5,
        }


def test_prepare_model_inputs_unwraps_the_audio():
    """The checked Audio reaches the backend as its bare waveform."""
    audio = Audio(np.ones(240, dtype=np.float32), 24000)
    out = CodecInferenceRunner.prepare_model_inputs({"audio": audio})
    assert isinstance(out["audio"], np.ndarray)
    assert out["audio"].shape == (240,)


def test_prepare_model_inputs_refuses_a_batch():
    """A list of waveforms is refused with the batch_size hint."""
    audio = Audio(np.ones(240, dtype=np.float32), 24000)
    with pytest.raises(ValueError, match="batch_size: null"):
        CodecInferenceRunner.prepare_model_inputs({"audio": [audio, audio]})


def test_format_output_keeps_only_the_waveform_as_float32():
    """Only `wav` survives, as a 1-D float32 array."""
    out = CodecInferenceRunner.format_output(
        {"codes": torch.zeros(1, 2, 3), "resyn_audio": torch.ones(1, 240)},
        batched=False,
    )
    assert list(out) == ["wav"]
    assert out["wav"].dtype == np.float32
    assert out["wav"].shape == (240,)


@pytest.mark.parametrize("result", [None, "wav", ["wav"]])
def test_format_output_rejects_a_non_mapping(result):
    """Anything but AudioCoding's dict is a type error."""
    with pytest.raises(TypeError, match="must return a dict"):
        CodecInferenceRunner.format_output(result, batched=False)


def test_format_output_requires_resyn_audio():
    """A result without `resyn_audio` is a key error."""
    with pytest.raises(KeyError, match="resyn_audio"):
        CodecInferenceRunner.format_output({"codes": torch.zeros(1)}, batched=False)


def test_format_output_refuses_a_batch():
    """A batched result is refused with the batch_size hint."""
    with pytest.raises(ValueError, match="batch_size: null"):
        CodecInferenceRunner.format_output([{"resyn_audio": 1}], batched=True)


def test_forward_runs_the_contract_around_the_backend():
    """The shared forward checks the input, calls the backend and checks the output."""
    model = FakeAudioCoding()
    stereo = np.stack([np.ones(480), np.ones(480)], axis=1).astype(np.float32)
    dataset = [{"utt_id": "u1", "audio": (48000, stereo)}]

    result = CodecInferenceRunner.forward(
        0,
        dataset=dataset,
        model=model,
        input_key="audio",
        inference_contract=dict(
            CONTRACT,
            inputs={
                "audio": {
                    "kind": "audio",
                    "sample_rate": 24000,
                    "channels": 1,
                    "normalize": ["mean_channels"],
                }
            },
        ),
    )

    # The contract resampled to 24 kHz and averaged the channels before the
    # backend saw a bare 1-D waveform.
    assert isinstance(model.seen, np.ndarray)
    assert model.seen.ndim == 1
    assert abs(model.seen.shape[0] - 240) <= 1
    assert isinstance(result["wav"], Audio)
    assert result["wav"].rate == 24000
    assert result["wav"].channels == 1
    assert result["utt_id"] == "u1"
