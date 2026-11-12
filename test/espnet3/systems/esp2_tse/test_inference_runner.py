"""Native TSE input and output adaptation through the shared boundary."""

import numpy as np
import pytest

from espnet3.contract.inference import load_inference_contract
from espnet3.contract.kinds.audio import Audio
from espnet3.systems.esp2_tse.src.inference_runner import ESP2TSEInferenceRunner


@pytest.fixture
def contract():
    return load_inference_contract(
        {
            "inputs": {
                name: {
                    "kind": "audio",
                    "sample_rate": 16000,
                    "channels": 1,
                    "rate_policy": "reject",
                }
                for name in ("speech_mix", "enroll_ref1")
            },
            "outputs": {"inf": {"kind": "audio", "sample_rate": 16000, "channels": 1}},
        }
    )


def test_native_target_and_rate(contract):
    samples = np.arange(128, dtype=np.float32) / 128

    def backend(speech_mix, enroll_ref1, fs):
        assert fs == 16000
        assert speech_mix.shape == enroll_ref1.shape == (128,)
        return [speech_mix.copy(), -speech_mix]

    result = ESP2TSEInferenceRunner.forward(
        0,
        dataset=[
            {
                "utt_id": "mix1",
                "speech_mix": Audio(samples, 16000),
                "enroll_ref1": Audio(samples, 16000),
            }
        ],
        model=backend,
        input_key=["speech_mix", "enroll_ref1"],
        inference_contract=contract,
    )
    assert result["utt_id"] == "mix1"
    assert result["inf"].rate == 16000
    np.testing.assert_array_equal(result["inf"].array, samples)


@pytest.mark.parametrize(
    "value", [Audio(np.zeros((2, 128)), 16000), Audio(np.zeros(128), 8000), 12]
)
def test_invalid_inputs_are_checked(contract, value):
    with pytest.raises((ValueError, TypeError)):
        ESP2TSEInferenceRunner.forward(
            0,
            dataset=[{"speech_mix": value, "enroll_ref1": value}],
            model=lambda **kwargs: pytest.fail("called backend"),
            input_key=["speech_mix", "enroll_ref1"],
            inference_contract=contract,
        )


def test_batch_rejected_before_backend(contract):
    item = {
        "speech_mix": Audio(np.zeros(128), 16000),
        "enroll_ref1": Audio(np.zeros(128), 16000),
    }
    with pytest.raises(ValueError, match="batch_size"):
        ESP2TSEInferenceRunner.forward(
            [0],
            dataset=[item],
            model=lambda **kwargs: pytest.fail("called backend"),
            input_key=list(item),
            inference_contract=contract,
        )
