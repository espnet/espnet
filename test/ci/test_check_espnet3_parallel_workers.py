"""ci/check_espnet3_parallel_workers.py on hand-made output directories.

The integration test trusts this checker to fail whenever the one- and
two-worker runs differ, and whenever the two-worker run was not split at
all. Both would otherwise go unnoticed, so its verdicts are pinned here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).parents[2] / "ci" / "check_espnet3_parallel_workers.py"
_spec = importlib.util.spec_from_file_location("check_parallel_workers", _SCRIPT)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


def _inference(root: Path, shards: int, extra_set: bool = False) -> Path:
    for name in ["test", "valid"] + (["extra"] if extra_set else []):
        d = root / name
        d.mkdir(parents=True)
        (d / "hyp.scp").write_text("utt0 a\nutt1 b\n")
        (d / "manifest.json").write_text(
            json.dumps({"shards": [{"shard_id": i} for i in range(shards)]})
        )
        for i in range(shards):
            (d / f"split.{i}").mkdir()
            (d / f"split.{i}" / "done").write_text("")
    (root / "metrics.json").write_text(json.dumps({"WER": {"test": 1.0}}))
    return root


def _stats(root: Path, shards: int, total: float = 10.0) -> Path:
    for mode in ["train", "valid"]:
        d = root / mode
        d.mkdir(parents=True)
        (d / "feats_shape").write_text("utt0 5,80\nutt1 7,80\n")
        np.savez(
            d / "feats_stats.npz",
            count=np.array(12),
            sum=np.full(80, total, dtype=np.float32),
            sum_square=np.full(80, total * 2, dtype=np.float32),
        )
        (d / "manifest.json").write_text(
            json.dumps({"shards": [{"shard_id": i} for i in range(shards)]})
        )
    return root


def test_identical_runs_split_in_two_pass(tmp_path):
    assert (
        checker.check_inference(
            _inference(tmp_path / "a", 1), _inference(tmp_path / "b", 2)
        )
        == []
    )
    assert (
        checker.check_stats(_stats(tmp_path / "c", 1), _stats(tmp_path / "d", 2)) == []
    )


def test_a_run_that_was_not_split_fails(tmp_path):
    failures = checker.check_inference(
        _inference(tmp_path / "a", 1), _inference(tmp_path / "b", 1)
    )
    assert any("not exercised" in f for f in failures)
    failures = checker.check_stats(_stats(tmp_path / "c", 1), _stats(tmp_path / "d", 1))
    assert any("not exercised" in f for f in failures)


def test_a_different_hypothesis_fails(tmp_path):
    serial, parallel = _inference(tmp_path / "a", 1), _inference(tmp_path / "b", 2)
    (parallel / "test" / "hyp.scp").write_text("utt0 a\nutt1 c\n")
    assert checker.check_inference(serial, parallel) == [
        "test/hyp.scp: differs between one and two workers"
    ]


@pytest.mark.parametrize("what", ["test set", "shape file", "statistics file"])
def test_an_output_only_the_parallel_run_wrote_fails(tmp_path, what):
    """Inventories are compared both ways, not only from the serial side."""
    if what == "test set":
        failures = checker.check_inference(
            _inference(tmp_path / "a", 1), _inference(tmp_path / "b", 2, extra_set=True)
        )
    else:
        serial, parallel = _stats(tmp_path / "c", 1), _stats(tmp_path / "d", 2)
        name = "extra_shape" if what == "shape file" else "extra_stats.npz"
        source = "feats_shape" if what == "shape file" else "feats_stats.npz"
        (parallel / "train" / name).write_bytes(
            (parallel / "train" / source).read_bytes()
        )
        failures = checker.check_stats(serial, parallel)
    assert any("one worker wrote" in f for f in failures), failures


def test_statistics_agree_to_rounding_but_not_beyond(tmp_path):
    serial = _stats(tmp_path / "a", 1, total=10.0)
    rounding = _stats(tmp_path / "b", 2, total=10.0 + 1e-6)
    assert checker.check_stats(serial, rounding) == []
    off = _stats(tmp_path / "c", 2, total=10.01)
    assert any("[sum]" in f for f in checker.check_stats(serial, off))
