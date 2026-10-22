"""Exercise the VOiCES overlay through the shared checked ASR runner."""

from pathlib import Path

import numpy as np
import pytest
from hydra.utils import get_class

from espnet3.contract.inference import load_inference_contract
from espnet3.utils.config_utils import load_and_merge_config

RECIPE = Path(__file__).resolve().parents[5] / "egs3/voices/esp2_asr"


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
    assert config.batch_size == 4
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


@pytest.mark.parametrize("condition", ["all", "source", "distant"])
@pytest.mark.parametrize("limit", [None, 3])
def test_filtered_reference_ids_match_batched_inference(
    corpus, inference_config, condition, limit
):
    """Materialize the same filtered rows and IDs that the batch runner decodes."""
    from egs3.voices.esp2_asr.dataset import Dataset, DatasetBuilder
    from espnet3.systems.base.inference_provider import InferenceProvider
    from espnet3.systems.base.metric import _dataset_column_scp

    recipe, _ = corpus
    DatasetBuilder().build(recipe_dir=recipe)
    config = inference_config
    config.recipe_dir = str(recipe)
    config.test_set = "test"
    entry = config.dataset.test[0]
    entry.data_src = "egs3.voices.esp2_asr.dataset"
    entry.data_src_args.update(
        split="valid", recipe_dir=str(recipe), condition=condition, limit=limit
    )
    dataset = InferenceProvider.build_dataset(config)
    full = Dataset("valid", recipe_dir=recipe)
    selected = [
        row
        for row in full.entries
        if condition == "all" or row["condition"] == condition
    ][:limit]
    assert len(dataset) == len(selected)
    references = _dataset_column_scp(
        config, recipe / "results", "test", "text", "utt_id"
    )
    ref_rows = dict(line.split(" ", 1) for line in references.read_text().splitlines())
    assert ref_rows == {str(i): row["text"] for i, row in enumerate(selected)}
    indices = list(reversed(range(len(dataset))))
    calls = []

    def backend(speech):
        calls.extend(speech)
        return [[(f"HYP {i}", [], [], None)] for i in indices]

    outputs = get_class(config.runner._target_).forward(
        indices,
        dataset=dataset,
        model=backend,
        input_key=config.input_key,
        inference_contract=config.contract,
    )
    assert len(calls) == len(indices)
    assert [out["utt_id"] for out in outputs] == [str(i) for i in indices]
    assert [(ref_rows[out["utt_id"]], out["text"]) for out in outputs] == [
        (selected[i]["text"], f"HYP {i}") for i in indices
    ]
