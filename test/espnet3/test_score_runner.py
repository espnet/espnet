import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet3.inference.abs_metrics import AbsMetrics
from espnet3.inference.score_runner import ScoreRunner

# ===============================================================
# Test Case Summary for ScoreRunner
# ===============================================================
#
# Valid Usage Tests
# | Test Name                         | Description                                                          | # noqa: E501
# |----------------------------------|----------------------------------------------------------------------| # noqa: E501
# | test_score_runner                | Full score evaluation with dummy WER/CER metrics                    | # noqa: E501
# | test_single_metric_apply_to_subset | Tests metric applied only to a subset of datasets                 | # noqa: E501
# | test_uid_sorting_in_load         | Confirms correct UID ordering when loading score files              | # noqa: E501
#
# Invalid Configuration/Error Handling Tests
# | Test Name                         | Description                                                          | # noqa: E501
# |----------------------------------|----------------------------------------------------------------------| # noqa: E501
# | test_missing_inputs_field_raises | Raises ValueError when 'inputs' field is missing                    | # noqa: E501
# | test_inputs_field_wrong_type     | Raises ValueError for wrong type in 'inputs' field                  | # noqa: E501
# | test_missing_target_field_raises | Raises KeyError when target field is missing in config              | # noqa: E501
# | test_metric_class_not_instance_of_absmetrics | Raises TypeError if metric is not AbsMetrics subclass   | # noqa: E501
# | test_uid_mismatch_raises         | Raises AssertionError if UID sets mismatch across inputs            | # noqa: E501
# | test_missing_hypothesis          | Raises AssertionError when hypothesis SCP file is missing           | # noqa: E501


# === Dummy metrics ===
class DummyWER(AbsMetrics):
    def __init__(self, inputs):
        self.ref_key, self.hyp_key = inputs

    def __call__(self, data, test_name, decode_dir):
        return {"wer": 0.1}


class DummyCER(AbsMetrics):
    def __init__(self, inputs):
        self.ref_key, self.hyp_key = inputs

    def __call__(self, data, test_name, decode_dir):
        return {"cer": 0.05}


class NotAMetric:
    def __init__(self, inputs):
        self.ref_key, self.hyp_key = inputs

    def __call__(self, data, test_name, decode_dir):
        return {"cer": 0.05}


# === Dummy Dataset ===
class DummyDataset:
    def __init__(self, path=None):
        self.data = [
            {"audio": np.random.random(16000), "text": "hello"},
            {"audio": np.random.random(16000), "text": "world"},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_config(name):
    config_path = f"test_utils/espnet3/config/{name}.yaml"
    return OmegaConf.load(config_path)


def copy_test_scores_to_tmp(tmp_path, src_dir="test_utils/espnet3/scores"):
    src_dir = Path(src_dir)
    for item in src_dir.glob("**/*"):
        if item.is_file():
            rel_path = item.relative_to(src_dir)
            dest = tmp_path / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)


# === valid test ===
def test_score_runner(tmp_path):
    # Copy test scores to temporary directory
    copy_test_scores_to_tmp(tmp_path)

    # Create a dummy ScoreRunner instance
    score_runner = ScoreRunner(
        load_config("test_score"),
        decode_dir=tmp_path,
    )

    # Run the score runner
    results = score_runner.run()

    # Check if results are as expected
    wer_result = results["test.espnet3.test_score_runner.DummyWER"]
    assert wer_result["test_1"]["wer"] == 0.1
    assert wer_result["test_2"]["wer"] == 0.1
    cer_result = results["test.espnet3.test_score_runner.DummyCER"]
    assert cer_result["test_1"]["cer"] == 0.05
    assert cer_result["test_2"]["cer"] == 0.05
    assert os.path.exists(tmp_path / "scores.json")


def test_single_metric_apply_to_subset(tmp_path):
    copy_test_scores_to_tmp(tmp_path)
    runner = ScoreRunner(load_config("test_score_single"), decode_dir=tmp_path)
    results = runner.run()

    assert "test.espnet3.test_score_runner.DummyWER" in results
    assert "test_1" in results["test.espnet3.test_score_runner.DummyWER"]
    assert "test_2" not in results["test.espnet3.test_score_runner.DummyWER"]


def test_uid_sorting_in_load(tmp_path):
    copy_test_scores_to_tmp(tmp_path)
    runner = ScoreRunner(load_config("test_score_unsorted"), decode_dir=tmp_path)
    results = runner.run()

    wer_result = results["test.espnet3.test_score_runner.DummyWER"]
    assert wer_result["test_unsorted"]["wer"] == 0.1


# === invalid tests ===
def test_missing_inputs_field_raises(tmp_path):
    copy_test_scores_to_tmp(tmp_path)
    with pytest.raises(ValueError):
        ScoreRunner(load_config("test_score_missing_inputs"), tmp_path)


def test_inputs_field_wrong_type(tmp_path):
    copy_test_scores_to_tmp(tmp_path)
    with pytest.raises(ValueError):
        ScoreRunner(load_config("test_score_invalid_inputs"), tmp_path)


def test_missing_target_field_raises(tmp_path):
    copy_test_scores_to_tmp(tmp_path)
    with pytest.raises(KeyError):
        ScoreRunner(load_config("test_score_missing_target"), tmp_path)


def test_metric_class_not_instance_of_absmetrics(tmp_path):
    copy_test_scores_to_tmp(tmp_path)
    runner = ScoreRunner(load_config("test_score_invalid_metric_class"), tmp_path)
    with pytest.raises(TypeError):
        runner.run()


def test_uid_mismatch_raises(tmp_path):
    copy_test_scores_to_tmp(tmp_path)
    with pytest.raises(AssertionError) as excinfo:
        ScoreRunner(load_config("test_score_uid_mismatch"), tmp_path)
    assert "Mismatch in UID sets across inputs" in str(excinfo.value)


def test_missing_hypothesis(tmp_path):
    copy_test_scores_to_tmp(tmp_path)
    with pytest.raises(AssertionError) as excinfo:
        ScoreRunner(load_config("test_score_missing_file"), tmp_path)
    assert "Missing SCP file" in str(excinfo.value)
