"""Unit tests for espnet3.systems.lace.src.inference_runner."""

import numpy as np
import pytest
import torch

from espnet3.contract.inference import load_inference_contract
from espnet3.contract.kinds.audio import Audio
from espnet3.systems.lace.src.inference_runner import LACEInferenceRunner

CONTRACT = load_inference_contract(
    {
        "inputs": {"audio": {"kind": "audio", "channels": 1}},
        "outputs": {"resyn_audio": {"kind": "audio", "channels": 1}},
    }
)


class _FakeCoding:
    """Stand-in backend: echoes the waveform back as ``resyn_audio``."""

    sample_rate = 120

    def __init__(self):
        """Record what the runner hands over."""
        self.seen = None

    def __call__(self, audio):
        """Return ``codes`` and the input as a tensor, as the real backend does."""
        self.seen = audio
        return {"codes": torch.zeros(1, 1, 3), "resyn_audio": torch.as_tensor(audio)}


def test_prepare_model_inputs_unwraps_the_checked_audio():
    """The backend receives the bare 1-D waveform of the contract's ``Audio``."""
    audio = Audio(np.zeros(240, dtype=np.float32), 120)

    out = LACEInferenceRunner.prepare_model_inputs({"audio": audio})

    assert isinstance(out["audio"], np.ndarray)
    assert out["audio"].shape == (240,)


def test_prepare_model_inputs_passes_a_bare_array_through():
    """An array that is already bare is left as it is."""
    array = np.zeros(240, dtype=np.float32)

    assert LACEInferenceRunner.prepare_model_inputs({"audio": array})["audio"] is array


def test_prepare_model_inputs_refuses_a_batch():
    """A list per key means ``batch_size`` was set; the runner says how to fix it."""
    audio = Audio(np.zeros(240, dtype=np.float32), 120)

    with pytest.raises(ValueError, match="batch_size: null"):
        LACEInferenceRunner.prepare_model_inputs({"audio": [audio, audio]})


def test_format_output_keeps_only_the_resynthesis_as_a_flat_float32_array():
    """``codes`` is dropped and the (1, T) tensor becomes a detached (T,) array."""
    result = {
        "codes": torch.zeros(2, 1, 60),
        "resyn_audio": torch.full((1, 240), 0.5, requires_grad=True),
    }

    out = LACEInferenceRunner.format_output(result, batched=False)

    assert list(out) == ["resyn_audio"]
    assert isinstance(out["resyn_audio"], np.ndarray)
    assert out["resyn_audio"].shape == (240,)
    assert out["resyn_audio"].dtype == np.float32
    assert float(out["resyn_audio"][0]) == 0.5


def test_format_output_refuses_a_batch_and_malformed_results():
    """Batched calls, non-dict results and a missing ``resyn_audio`` raise."""
    with pytest.raises(ValueError, match="batch_size: null"):
        LACEInferenceRunner.format_output([{}], batched=True)
    with pytest.raises(TypeError, match="must return a dict"):
        LACEInferenceRunner.format_output(torch.zeros(4), batched=False)
    with pytest.raises(KeyError, match="resyn_audio"):
        LACEInferenceRunner.format_output({"codes": torch.zeros(1)}, batched=False)


def test_forward_runs_the_contract_around_the_backend():
    """Through ``forward`` the item is checked, resampled, unwrapped and stamped.

    The dataset holds a stereo ``(rate, samples)`` pair at 240 Hz; the contract
    averages it to mono and resamples it to the backend's 120 Hz before the
    backend sees a bare array, and the returned ``resyn_audio`` is an ``Audio``
    at the backend's rate with the item's ``utt_id`` attached.
    """
    contract = load_inference_contract(
        {
            "inputs": {
                "audio": {
                    "kind": "audio",
                    "channels": 1,
                    "normalize": ["mean_channels"],
                }
            },
            "outputs": {"resyn_audio": {"kind": "audio", "channels": 1}},
        }
    )
    backend = _FakeCoding()
    stereo = np.stack([np.ones(480), -np.ones(480)]).astype(np.float32)
    dataset = [{"utt_id": "u1", "audio": (240, stereo)}]

    out = LACEInferenceRunner.forward(
        0,
        dataset=dataset,
        model=backend,
        input_key="audio",
        inference_contract=contract,
    )

    assert isinstance(backend.seen, np.ndarray) and backend.seen.ndim == 1
    assert backend.seen.shape == (240,)  # 480 samples at 240 Hz -> 120 Hz
    assert out["utt_id"] == "u1"
    assert isinstance(out["resyn_audio"], Audio)
    assert out["resyn_audio"].rate == 120
    assert out["resyn_audio"].channels == 1


def test_forward_refuses_an_undeclared_input_key():
    """The runner refuses an ``input_key`` the contract does not declare."""
    with pytest.raises(ValueError, match="input_key must match"):
        LACEInferenceRunner.forward(
            0,
            dataset=[{"utt_id": "u1", "speech": np.zeros(120, dtype=np.float32)}],
            model=_FakeCoding(),
            input_key="speech",
            inference_contract=CONTRACT,
        )
