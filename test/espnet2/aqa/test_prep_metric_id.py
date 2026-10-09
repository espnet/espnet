"""Test the prep_metric_id utility."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / "egs2/TEMPLATE/asr1/pyscripts/utils"


def test_metric_reading_limit(tmp_path):
    """Stop discovery before decoding rows past the requested limit."""
    path = tmp_path / "metric.scp"
    path.write_text('a {"mos": 1}\nb invalid-json\n')
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


@pytest.mark.parametrize("use_types", [True, False])
def test_metric_discovery(tmp_path, use_types):
    """Discover each metric once or use an explicitly supplied vocabulary."""
    path = tmp_path / "metric.scp"
    path.write_text('a {"mos": 1}\nb {"mos": 2, "wer": 3}\n')
    output = tmp_path / "metric2id"
    options = []
    if use_types:
        mapping = tmp_path / "metric2type"
        mapping.write_text("quality numeric\nlanguage categorical\n")
        options = ["--metric2type", str(mapping)]
    subprocess.run(
        [
            sys.executable,
            str(UTILS / "prep_metric_id.py"),
            str(path),
            str(output),
            *options,
        ],
        check=True,
    )
    assert output.read_text().splitlines() == (
        ["quality", "language"] if use_types else ["mos", "wer"]
    )
