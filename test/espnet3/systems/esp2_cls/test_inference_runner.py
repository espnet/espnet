"""Tests for the classification inference runner."""

import numpy as np
import pytest
import torch

from espnet3.contract.inference import load_inference_contract
from espnet3.systems.esp2_cls.src.inference_runner import ESP2CLSInferenceRunner

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_forward_writes_hyp_and_score           | The contracted outputs carry |
# |                                             | the label and every score.   |
# | test_format_output_rejects_a_batch          | Classification has no batch  |
# |                                             | call.                        |

CONTRACT = {
    "inputs": {"speech": {"kind": "audio", "sample_rate": 16000, "channels": 1}},
    "outputs": {"hyp": {"kind": "text"}, "score": {"kind": "text"}},
}


def test_forward_writes_hyp_and_score():
    seen = {}

    def model(speech):
        seen["speech"] = speech
        return [1], torch.tensor([0.25, 0.75]), "joy"

    out = ESP2CLSInferenceRunner.forward(
        0,
        dataset=[{"speech": np.zeros(1600, dtype=np.float32), "label": "joy"}],
        model=model,
        input_key="speech",
        inference_contract=load_inference_contract(CONTRACT),
    )

    assert out == {"hyp": "joy", "score": "0.25 0.75", "utt_id": "0"}
    assert isinstance(seen["speech"], np.ndarray)
    assert seen["speech"].shape == (1600,)


def test_format_output_rejects_a_batch():
    with pytest.raises(ValueError, match="batch_size: null"):
        ESP2CLSInferenceRunner.format_output([], batched=True)
