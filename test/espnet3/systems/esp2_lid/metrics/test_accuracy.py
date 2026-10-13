import json
from pathlib import Path

import pytest

from espnet3.contract.metrics import MetricContractError
from espnet3.systems.esp2_lid.metrics.accuracy import Accuracy
from espnet3.systems.esp2_lid.metrics.classification_report import ClassificationReport


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    ref = tmp_path / "ref.scp"
    hyp = tmp_path / "hyp.scp"
    ref.write_text("utt1 eng\nutt2 eng\nutt3 fra\nutt4 deu\n", encoding="utf-8")
    hyp.write_text("utt1 eng\nutt2 fra\nutt3 fra\nutt4 eng\n", encoding="utf-8")
    return {"ref": ref, "hyp": hyp}


def test_accuracy_matches_voxlingua_scoring(tmp_path: Path):
    data = _write_inputs(tmp_path)

    result = Accuracy()(data, "dev", tmp_path)

    assert result == {"MicroAccuracy": 50.0, "MacroAccuracy": 50.0}
    assert list((tmp_path / "dev").iterdir()) == []  # Accuracy writes no files
    result.update(ClassificationReport()(data, "dev", tmp_path))
    assert result == {
        "MicroAccuracy": 50.0,
        "MacroAccuracy": 50.0,
        "MacroPrecision": 33.33,
        "MacroRecall": 50.0,
        "MacroF1": 38.89,
    }
    per_language = json.loads((tmp_path / "dev/lid_per_language.json").read_text())
    assert per_language["eng"] == {
        "Count": 2,
        "Correct": 1,
        "Accuracy": 50.0,
        "Precision": 50.0,
        "Recall": 50.0,
        "F1": 50.0,
    }
    assert per_language["fra"]["Precision"] == 50.0
    assert per_language["deu"]["Recall"] == 0.0
    assert json.loads((tmp_path / "dev/lid_error_counts.json").read_text()) == {
        "deu->eng": 1,
        "eng->fra": 1,
    }


@pytest.mark.parametrize("metric_class", [Accuracy, ClassificationReport])
def test_merged_batches_score_like_one_pass(tmp_path: Path, metric_class):
    """Chunks and ranks add counts, so their merge equals reading everything."""
    data = _write_inputs(tmp_path)
    expected = metric_class()(data, "dev")

    first, second = metric_class(), metric_class()
    first.update(ref=["eng", "eng"], hyp=["eng", "fra"])
    second.update(ref=["fra", "deu"], hyp=["fra", "eng"])
    merged = metric_class()
    merged.load_state(merged.merge_states([first.state(), second.state()]))

    assert merged.compute() == expected


@pytest.mark.parametrize("metric_class", [Accuracy, ClassificationReport])
def test_lid_metrics_reject_empty_and_missing_labels(metric_class):
    metric = metric_class()
    with pytest.raises(ValueError, match="No LID predictions"):
        metric.compute()
    with pytest.raises(ValueError, match="Empty LID label"):
        metric.update(ref=["eng"], hyp=[" "])
    with pytest.raises(MetricContractError):
        metric.update(ref=["eng"])
