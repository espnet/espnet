from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from espnet3.contract.inference import load_inference_contract
from espnet3.systems.base.inference_runner import InferenceRunner
from espnet3.systems.esp2_enh.src.inference import Inference

TEMPLATE_INFERENCE = (
    Path(__file__).resolve().parents[4]
    / "egs3"
    / "TEMPLATE"
    / "esp2_enh"
    / "conf"
    / "inference.yaml"
)


class FakeSeparateSpeech:
    """Returns twice the input as speaker 1 and silence as speaker 2."""

    def __init__(self):
        self.calls = []

    def __call__(self, speech_mix, fs):
        self.calls.append((speech_mix.copy(), fs))
        return [speech_mix * 2.0, np.zeros_like(speech_mix)]


def _infer(model, dataset, idx):
    """Run the shared runner with the TEMPLATE input_key and contract."""
    config = OmegaConf.load(TEMPLATE_INFERENCE)
    return InferenceRunner.forward(
        idx,
        dataset=dataset,
        model=model,
        input_key=config.input_key,
        inference_contract=load_inference_contract(config.contract),
    )


def test_batch_is_one_padded_call_and_each_output_keeps_its_length():
    backend = FakeSeparateSpeech()
    dataset = [
        {"utt_id": "a", "speech_mix": (16000, np.full(800, 0.1, dtype=np.float32))},
        {"utt_id": "b", "speech_mix": (16000, np.full(400, 0.2, dtype=np.float32))},
    ]

    out = _infer(Inference(backend), dataset, [0, 1])

    assert len(backend.calls) == 1
    batch, fs = backend.calls[0]
    assert batch.shape == (2, 800) and fs == 16000
    assert not batch[1, 400:].any()
    assert [o["utt_id"] for o in out] == ["a", "b"]
    assert [o["enhanced"].array.shape for o in out] == [(800,), (400,)]
    np.testing.assert_allclose(out[0]["enhanced"].array, 0.2)
    np.testing.assert_allclose(out[1]["enhanced"].array, 0.4)
    assert out[0]["enhanced"].rate == 16000


def test_output_that_would_clip_is_scaled_to_a_0_9_peak():
    wave = np.array([0.1, -0.75, 0.5], dtype=np.float32)

    out = _infer(Inference(FakeSeparateSpeech()), [{"speech_mix": (16000, wave)}], 0)

    np.testing.assert_allclose(out["enhanced"].array, [0.12, -0.9, 0.6], rtol=1e-6)


def test_input_is_resampled_to_16k_and_only_its_first_channel_is_enhanced():
    backend = FakeSeparateSpeech()
    stereo = np.stack(
        [np.full(800, 0.1, dtype=np.float32), np.full(800, 0.3, dtype=np.float32)]
    )

    out = _infer(Inference(backend), [{"speech_mix": (8000, stereo)}], 0)

    batch, fs = backend.calls[0]
    assert batch.shape == (1, 1600) and fs == 16000
    np.testing.assert_allclose(batch[0, 100:-100], 0.1, atol=1e-3)
    assert out["enhanced"].rate == 16000
