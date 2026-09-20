import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / "egs2/TEMPLATE/asr1/pyscripts/utils"
spec = importlib.util.spec_from_file_location(
    "universa_eval", UTILS / "universa_eval.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_metric_union(tmp_path):
    path = tmp_path / "metric.scp"
    path.write_text('a {"mos": 1}\nb {"wer": 2}\n')
    _, names = evaluation.load_metrics(path, True)
    assert names == {"mos", "wer"}


def test_metric_reading_limit(tmp_path):
    path = tmp_path / "metric.scp"
    path.write_text('a {"mos": 1}\nb {"wer": 2}\n')
    output = tmp_path / "metric2id"
    subprocess.run(
        [
            sys.executable,
            str(UTILS / "prep_metric_id.py"),
            str(path),
            str(output),
            "--reading_size",
            "1",
        ],
        check=True,
    )
    assert output.read_text() == "mos\n"


@pytest.mark.parametrize("skip_missing", ["true", "false"])
def test_missing_utterance(tmp_path, skip_missing):
    reference = tmp_path / "ref.scp"
    prediction = tmp_path / "pred.scp"
    output = tmp_path / "result.json"
    reference.write_text('a {"mos": 1}\nb {"mos": 2}\n')
    prediction.write_text('a {"mos": 1}\nb {"mos": 2}\nc {"mos": 3}\n')
    result = subprocess.run(
        [
            sys.executable,
            str(UTILS / "universa_eval.py"),
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
