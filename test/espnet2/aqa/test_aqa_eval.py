"""Test the aqa_eval utility."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / "egs2/TEMPLATE/asr1/pyscripts/utils"
spec = importlib.util.spec_from_file_location("aqa_eval", UTILS / "aqa_eval.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_metric_union(tmp_path):
    """Discover metric names across all rows."""
    path = tmp_path / "metric.scp"
    path.write_text('a {"mos": 1}\nb {"wer": 2}\n')
    _, names = evaluation.load_metrics(path, True)
    assert names == {"mos", "wer"}


@pytest.mark.parametrize("skip_missing", ["true", "false"])
def test_missing_utterance(tmp_path, skip_missing):
    """Respect strict and permissive handling of prediction-only utterances."""
    reference = tmp_path / "ref.scp"
    prediction = tmp_path / "pred.scp"
    output = tmp_path / "result.json"
    reference.write_text('a {"mos": 1}\nb {"mos": 2}\n')
    prediction.write_text('a {"mos": 1}\nb {"mos": 2}\nc {"mos": 3}\n')
    result = subprocess.run(
        [
            sys.executable,
            str(UTILS / "aqa_eval.py"),
            "--ref_metrics",
            str(reference),
            "--pred_metrics",
            str(prediction),
            "--out_file",
            str(output),
            "--skip_missing",
            skip_missing,
        ],
        capture_output=True,
        text=True,
    )
    if skip_missing == "true":
        assert result.returncode == 0, result.stderr
        assert json.loads(output.read_text())["utt_mos_mse"] == 0
    else:
        assert result.returncode != 0
        assert "Missing utterance: c" in result.stderr


def run_evaluation(tmp_path, reference, prediction, *options):
    """Run the public evaluation CLI with small score files."""
    ref = tmp_path / "ref.scp"
    pred = tmp_path / "pred.scp"
    output = tmp_path / "result.json"
    ref.write_text(reference)
    pred.write_text(prediction)
    result = subprocess.run(
        [
            sys.executable,
            str(UTILS / "aqa_eval.py"),
            "--ref_metrics",
            str(ref),
            "--pred_metrics",
            str(pred),
            "--out_file",
            str(output),
            *options,
        ],
        capture_output=True,
        text=True,
    )
    return result, output


@pytest.mark.parametrize("prediction", ['a {"mos": 1}\n', "a {}\nb {}\n"])
def test_missing_predictions_rejected(tmp_path, prediction):
    """Strict evaluation must detect missing predicted rows and metrics."""
    result, _ = run_evaluation(tmp_path, 'a {"mos": 1}\nb {"mos": 2}\n', prediction)
    assert result.returncode != 0
    assert "Missing" in result.stderr


@pytest.mark.parametrize(
    "reference,prediction",
    [
        ('a {"mos": 1}\n', 'b {"mos": 2}\n'),
        ('a {"mos": 1}\n', 'a {"wer": 2}\n'),
        ("", ""),
    ],
)
def test_no_matching_scores(tmp_path, reference, prediction):
    """Skipping absent scores must not produce a successful empty evaluation."""
    result, output = run_evaluation(
        tmp_path, reference, prediction, "--skip_missing", "true"
    )
    assert result.returncode != 0
    assert "No matching" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("scores", [[1], [1, 1]])
def test_undefined_correlations_are_null(tmp_path, scores):
    """Singleton and constant scores retain MSE and serialize strict JSON."""
    rows = "".join(f'{i} {{"mos": {value}}}\n' for i, value in enumerate(scores))
    result, output = run_evaluation(tmp_path, rows, rows)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == {
        "utt_mos_mse": 0,
        "utt_mos_lcc": None,
        "utt_mos_srcc": None,
        "utt_mos_ktau": None,
    }


def test_system_averages(tmp_path):
    """System scores must average matched utterances before computing errors."""
    mapping = tmp_path / "utt2sys"
    mapping.write_text("a first\nb first\nc second\n")
    result, output = run_evaluation(
        tmp_path,
        'a {"mos": 1}\nb {"mos": 3}\nc {"mos": 4}\n',
        'a {"mos": 2}\nb {"mos": 4}\nc {"mos": 6}\n',
        "--level",
        "sys",
        "--sys_info",
        str(mapping),
    )
    assert result.returncode == 0, result.stderr
    scores = json.loads(output.read_text())
    assert scores["sys_mos_mse"] == pytest.approx(2.5)
    assert scores["sys_mos_lcc"] == pytest.approx(1)


@pytest.mark.parametrize(
    "bad_rows",
    [
        "a {}\na {}\n",
        "a []\n",
        "a null\n",
        "a\n",
    ],
)
def test_invalid_metric_rows(tmp_path, bad_rows):
    """Reject malformed or duplicate score records instead of overwriting them."""
    path = tmp_path / "metric.scp"
    path.write_text(bad_rows)
    with pytest.raises(ValueError):
        evaluation.load_metrics(path, True)


@pytest.mark.parametrize("value", ['"good"', "null", "true", "NaN", "[1]"])
def test_invalid_numeric_score(tmp_path, value):
    """Give a clear error for nonnumeric and nonfinite evaluation scores."""
    result, _ = run_evaluation(tmp_path, 'a {"mos": 1}\n', f'a {{"mos": {value}}}\n')
    assert result.returncode != 0
    assert "finite numeric scalar" in result.stderr


def test_partial_metric_overlap(tmp_path):
    """Only paired scores contribute when metrics have different coverage."""
    result, output = run_evaluation(
        tmp_path,
        'a {"mos": 1, "wer": 10}\nb {"mos": 3}\nc {"mos": 9}\n',
        'a {"mos": 2}\nb {"mos": 5, "wer": 20}\n',
        "--skip_missing",
        "true",
    )
    assert result.returncode == 0, result.stderr
    scores = json.loads(output.read_text())
    assert scores["utt_mos_mse"] == pytest.approx(2.5)
    assert not any("wer" in key for key in scores)


@pytest.mark.parametrize("skip_missing", ["true", "false"])
def test_missing_system_mapping(tmp_path, skip_missing):
    """System mapping gaps follow the selected missing-score policy."""
    mapping = tmp_path / "utt2sys"
    mapping.write_text("a first\n")
    rows = 'a {"mos": 1}\nb {"mos": 2}\n'
    result, output = run_evaluation(
        tmp_path,
        rows,
        rows,
        "--level",
        "sys",
        "--sys_info",
        str(mapping),
        "--skip_missing",
        skip_missing,
    )
    if skip_missing == "true":
        assert result.returncode == 0, result.stderr
        assert json.loads(output.read_text())["sys_mos_mse"] == 0
    else:
        assert result.returncode != 0
        assert "Missing system information" in result.stderr
