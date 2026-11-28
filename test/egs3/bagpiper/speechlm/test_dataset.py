import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from hydra.utils import instantiate

from egs3.bagpiper.speechlm.src.requests import Dataset
from espnet3.contract.inference import load_inference_contract
from espnet3.systems.base.inference_runner import InferenceRunner
from espnet3.utils.config_utils import load_and_merge_config


def write_split(directory, samples, rows):
    manifest = directory / "valid.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": samples,
                "data_entry": [
                    {
                        "name": "dialogue",
                        "reader": "dialogue",
                        "path": "dialogues.jsonl",
                    }
                ],
            }
        )
    )
    (directory / "dialogues.jsonl").write_text(
        "".join(
            json.dumps({"example_id": key, "messages": turns}) + "\n"
            for key, turns in rows
        )
    )
    return manifest


def test_rerunning_inference_replaces_output_for_the_same_ids(tmp_path):
    config = load_and_merge_config(
        Path(__file__).resolve().parents[4]
        / "egs3/bagpiper/speechlm/conf/inference.yaml",
        "inference.yaml",
    )
    model = Mock(side_effect=[{"text": "first"}, {"text": "second"}])
    provider = Mock()
    provider.build_env_local.return_value = {
        "dataset": [{"utt_id": "one", "prompt": "question"}],
        "model": model,
        "input_key": "prompt",
        "inference_contract": {
            "inputs": {"prompt": {"kind": "text"}},
            "outputs": {"text": {"kind": "text"}},
        },
    }
    for expected in ("first", "second"):
        runner = instantiate(config.runner, provider=provider, output_dir=tmp_path)
        assert runner([0])
        assert (tmp_path / "text.scp").read_text() == f"one {expected}\n"


def test_audio_understanding_preserves_order_and_hides_answers(tmp_path):
    question = [["user", "audio", "clip.wav"], ["user", "text", "What was said?"]]
    answer = [["assistant", "text", "private answer"], ["user", "text", "next turn"]]
    manifest = write_split(
        tmp_path,
        ["b", "a"],
        [
            ("outside", []),
            ("a", [["system", "text", "Listen first."], *question, *answer]),
            ("b", question),
        ],
    )
    data = Dataset(manifest, "understand")
    assert len(data) == 2
    first = {
        "utt_id": "b",
        "prompt": "What was said?",
        "speech": str(tmp_path / "clip.wav"),
    }
    assert data[0] == first
    assert data[1] == {**first, "utt_id": "a", "system": "Listen first."}


def test_audio_generation_does_not_read_target_waveform(tmp_path):
    manifest = write_split(
        tmp_path,
        ["a"],
        [
            (
                "a",
                [
                    ["user", "text", "Make a sound"],
                    ["assistant", "audio", "missing.wav"],
                ],
            )
        ],
    )
    assert Dataset(manifest, "generate")[0] == {"utt_id": "a", "prompt": "Make a sound"}
    with pytest.raises(ValueError, match="Unexpected understand request turns"):
        Dataset(manifest, "understand")


def test_incomplete_split_reports_missing_dialogue(tmp_path):
    manifest = write_split(tmp_path, ["missing"], [])
    with pytest.raises(ValueError, match="Missing 1 dialogue"):
        Dataset(manifest, "generate")


@pytest.mark.parametrize("bad_turn", [["user", "text"], ["user", "text", 42], None])
def test_malformed_input_turn_names_the_dialogue(tmp_path, bad_turn):
    manifest = write_split(
        tmp_path,
        ["good", "broken"],
        [("good", [["user", "text", "Listen"]]), ("broken", [bad_turn])],
    )
    with pytest.raises(ValueError, match="Invalid request turn for 'broken'"):
        Dataset(manifest, "generate")


def test_invalid_id_fails_before_inference(tmp_path):
    manifest = write_split(tmp_path, ["two words"], [])
    with pytest.raises(ValueError, match="two words.*SCP"):
        Dataset(manifest, "generate")


def test_relative_audio_path_survives_working_directory_change(tmp_path, monkeypatch):
    recipe = tmp_path / "recipe"
    recipe.mkdir()
    monkeypatch.chdir(recipe)
    question = [["user", "audio", "clip.wav"], ["user", "text", "Listen"]]
    manifest = write_split(Path("."), ["one"], [("one", question)])
    sf.write("clip.wav", np.ones(16, dtype=np.float32), 16000)
    data = Dataset(manifest, "understand")
    monkeypatch.chdir(tmp_path)
    assert sf.info(data[0]["speech"]).frames == 16


def test_understanding_config_handles_absent_and_empty_system_turn(tmp_path):
    sf.write(tmp_path / "clip.wav", np.zeros(8, dtype=np.float32), 8000)
    question = [["user", "audio", "clip.wav"], ["user", "text", "Listen"]]
    manifest = write_split(
        tmp_path,
        ["without", "empty"],
        [("without", question), ("empty", [["system", "text", ""], *question])],
    )
    recipe = Path(__file__).resolve().parents[4] / "egs3/bagpiper/speechlm"
    config = load_and_merge_config(
        recipe / "conf/inference.yaml",
        config_name="inference.yaml",
        default_package="egs3.TEMPLATE.speechlm",
    )
    contract = load_inference_contract(config.contract)
    assert config.batch_size is None
    data = Dataset(manifest, "understand")
    called = []

    def understand(**inputs):
        called.append(inputs)
        return {"text": "answer"}

    understand.sample_rate = None
    for idx, key in enumerate(("without", "empty")):
        result = InferenceRunner.forward(
            idx,
            data,
            understand,
            input_key=config.input_key,
            inference_contract=contract,
        )
        assert result == {"utt_id": key, "text": "answer"}
    assert "system" not in called[0]
    assert called[1]["system"] == ""
    assert called[0]["speech"].rate == 8000
