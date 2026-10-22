"""Exercise the AN4 overlay through the shared checked ASR runner."""

from pathlib import Path

import numpy as np
import pytest
from hydra.utils import get_class

from espnet3.contract.inference import load_inference_contract
from espnet3.utils.config_utils import load_and_merge_config

RECIPE = Path(__file__).resolve().parents[5] / "egs3/an4/esp2_asr"


@pytest.fixture
def inference_config():
    """Resolve the recipe overlay against the installed ASR template."""
    return load_and_merge_config(RECIPE / "conf/inference.yaml", "inference.yaml")


def test_shared_runner_keeps_indices_and_excludes_references(inference_config):
    """The backend receives only speech; text outputs retain dataset order."""
    config = inference_config
    runner = get_class(config.runner._target_)
    contract = load_inference_contract(config.contract)
    assert config.input_key == "speech"
    assert "output_fn" not in config
    assert config.batch_size is None
    samples = [
        {"speech": np.ones(160, dtype=np.float32), "text": "REFERENCE A"},
        {"speech": np.zeros(80, dtype=np.float32), "text": "REFERENCE B"},
    ]

    def backend(speech):
        return [("WORD" if speech.any() else None, [], [], None)]

    outputs = [
        runner.forward(
            i,
            dataset=samples,
            model=backend,
            input_key=config.input_key,
            inference_contract=contract,
        )
        for i in [1, 0]
    ]
    assert outputs == [{"text": "", "utt_id": "1"}, {"text": "WORD", "utt_id": "0"}]
    assert samples[0]["text"] == "REFERENCE A"


@pytest.mark.parametrize(
    "speech",
    [
        None,
        123,
        np.zeros((2, 160), dtype=np.float32),
        np.zeros((1, 2, 3), dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
    ],
)
def test_recipe_contract_rejects_invalid_audio(inference_config, speech):
    """The inherited mono/finite-audio boundary rejects inputs before decoding."""
    calls = []
    config = inference_config
    runner = get_class(config.runner._target_)
    with pytest.raises((TypeError, ValueError)):
        runner.forward(
            0,
            dataset=[{"speech": speech}],
            model=lambda speech: calls.append(speech),
            input_key=config.input_key,
            inference_contract=config.contract,
        )
    assert calls == []


def test_known_rate_uses_shared_resampling_policy(inference_config):
    """The template resamples known-rate mono audio to its explicit 16 kHz."""
    config = inference_config
    contract = load_inference_contract(config.contract)
    checked = contract.prepare_input(
        {"speech": (8000, np.zeros(80, dtype=np.float32))},
        model=None,
    )
    assert checked["speech"].rate == 16000
    assert checked["speech"].array.shape == (160,)
