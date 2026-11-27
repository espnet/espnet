"""Tests for ``espnet3.bin.atlas.worker``."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from espnet3.bin.atlas import worker


class _Runtime:
    def __init__(self, execution, status):
        self.graph = SimpleNamespace(nodes={"n": SimpleNamespace(execution=execution)})
        self.status = status
        self.calls = []

    def run_gpu_self_controller(self, **kwargs):
        self.calls.append(("gpu", kwargs))
        return SimpleNamespace(status=self.status)

    def run_node(self, **kwargs):
        self.calls.append(("node", kwargs))
        return SimpleNamespace(status=self.status)


def _argv(tmp_path, *extra):
    return [
        "worker",
        "--study-dir",
        str(tmp_path),
        "--recipe-dir",
        str(tmp_path),
        "--node",
        "n",
        "--run-id",
        "run_1",
        *extra,
    ]


@pytest.mark.parametrize(
    "execution, status, code, kind",
    [
        ("submission", "success", 0, "gpu"),
        ("submission", "failure", 1, "gpu"),
        ("local", "terminal", 0, "node"),
        ("local", "timeout", 1, "node"),
    ],
)
def test_main_runs_node_and_maps_status_to_exit_code(
    tmp_path, monkeypatch, execution, status, code, kind
):
    """Dispatch by execution mode and fail on failure/timeout results."""
    runtime = _Runtime(execution, status)
    monkeypatch.setattr(worker, "_load_runtime", lambda study, recipe: runtime)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--trial-id", "t1"))

    assert worker.main() == code

    ((called, kwargs),) = runtime.calls
    assert called == kind
    assert kwargs["node_name"] == "n" and kwargs["run_id"] == "run_1"
    assert kwargs["trial_id"] == "t1" and kwargs["attempt_id"] == "attempt_000000"
    if kind == "node":
        assert kwargs["controller_mode"] is False


def test_main_requires_study_arguments(monkeypatch):
    """Exit with a usage error when required arguments are missing."""
    monkeypatch.setattr(sys, "argv", ["worker"])
    with pytest.raises(SystemExit):
        worker.main()
