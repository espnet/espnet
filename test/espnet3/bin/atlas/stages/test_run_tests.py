"""Tests for espnet3.bin.atlas.stages.run_tests."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from espnet3.bin.atlas.core.models import Trial
from espnet3.bin.atlas.core.state import StateStore
from espnet3.bin.atlas.stages import run_tests as run_tests_mod
from espnet3.bin.atlas.stages.run_tests import RunTestsStage

STUDY = "study"


def load_yaml(path):
    """Load a YAML file as plain containers."""
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _trial(trial_id="trial_000001", score=None, score_name="wer", **kwargs):
    """Build a minimal trial."""
    now = datetime.now(timezone.utc).isoformat()
    fields = dict(
        trial_id=trial_id,
        study_id=STUDY,
        status="proposed",
        config_patch={},
        resolved_config_path=None,
        rationale="",
        expected_effect="",
        risk="",
        score=score,
        score_name=score_name,
        decision=None,
        parent_trial_id=None,
        attempt_count=0,
        created_at=now,
        updated_at=now,
    )
    fields.update(kwargs)
    return Trial(**fields)


def _context(tmp_path, *, trial=None, atlas=None, agent=None):
    """Build a StageContext stand-in backed by a real StateStore."""
    study_dir = tmp_path / "study"
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir(exist_ok=True)
    state = StateStore(study_dir)
    state.create_or_load_study(STUDY, study_dir)
    if trial is not None:
        state.create_trial(trial)
    trial_dir = study_dir / "trials" / (trial.trial_id if trial else "none")
    trial_dir.mkdir(parents=True, exist_ok=True)
    base = {"metric": {"name": "wer", "mode": "min", "extraction": []}}
    base.update(atlas or {})
    return SimpleNamespace(
        config=OmegaConf.create({"atlas": base}),
        study_dir=study_dir,
        recipe_dir=recipe_dir,
        current_trial=trial,
        state=state,
        agent=agent,
        scheduler=SimpleNamespace(study_id=STUDY),
        logger=logging.getLogger("test_atlas"),
        get_trial_dir=lambda: trial_dir,
        get_artifact_dir=lambda: study_dir / "node_runs" / "run",
    )


def _fake_pytest(monkeypatch, returncode, stdout="out", stderr=""):
    """Replace subprocess.run in run_tests and record the calls."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    monkeypatch.setattr(run_tests_mod.subprocess, "run", fake_run)
    return calls


def _code_edit(**kwargs):
    return {"code_edit": kwargs}


def test_create_next_run_dir_numbers_runs_sequentially(tmp_path):
    first, n1 = run_tests_mod._create_next_run_dir(tmp_path)
    second, n2 = run_tests_mod._create_next_run_dir(tmp_path)
    assert (first.name, n1, second.name, n2) == ("run_001", 1, "run_002", 2)


def test_find_test_dirs_prefers_tests_subdir_then_test_files(tmp_path):
    (tmp_path / "a" / "tests").mkdir(parents=True)
    (tmp_path / "b" / "pkg").mkdir(parents=True)
    (tmp_path / "b" / "pkg" / "test_x.py").write_text("", encoding="utf-8")
    (tmp_path / "b" / "pkg" / "util.py").write_text("", encoding="utf-8")
    found = run_tests_mod._find_test_dirs(tmp_path, ["a", "b", "missing"])
    assert found == [tmp_path / "a" / "tests", tmp_path / "b" / "pkg" / "test_x.py"]


def test_find_test_dirs_falls_back_to_the_allowed_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    found = run_tests_mod._find_test_dirs(tmp_path, ["src", "gone"])
    assert found == [tmp_path / "src", tmp_path / "gone"]


def test_run_tests_succeeds_and_logs_the_pytest_run(tmp_path, monkeypatch):
    calls = _fake_pytest(monkeypatch, 0, stdout="2 passed")
    context = _context(tmp_path, trial=_trial(), atlas=_code_edit(allowed_dirs=["src"]))
    (context.recipe_dir / "src" / "tests").mkdir(parents=True)

    result = RunTestsStage().run(context)

    assert result.status == "success"
    assert result.payload["run_index"] == 1
    assert Path(result.artifacts["pytest_log"]).read_text() == "2 passed"
    cmd, kwargs = calls[0]
    assert cmd[-1] == str(context.recipe_dir / "src" / "tests")
    assert kwargs["cwd"] == str(context.recipe_dir)
    run_dir = context.get_trial_dir() / "test_runs" / "run_001"
    meta = load_yaml(run_dir / "meta.yaml")
    assert meta["returncode"] == 0 and meta["trial_id"] == "trial_000001"
    assert not (run_dir / "error.log").exists()


def test_run_tests_failure_writes_error_log_including_stderr(tmp_path, monkeypatch):
    _fake_pytest(monkeypatch, 1, stdout="1 failed", stderr="boom")
    context = _context(tmp_path, trial=_trial(), atlas=_code_edit(allowed_dirs=["src"]))

    result = RunTestsStage().run(context)

    assert result.status == "failure"
    assert "run 1/5" in result.message
    error = Path(result.artifacts["error_log"]).read_text(encoding="utf-8")
    assert error == "1 failed\nboom"


def test_run_tests_gives_up_after_max_test_retries(tmp_path, monkeypatch):
    calls = _fake_pytest(monkeypatch, 1)
    context = _context(
        tmp_path,
        trial=_trial(),
        atlas=_code_edit(allowed_dirs=["src"], max_test_retries=2),
    )
    stage = RunTestsStage()

    statuses = [stage.run(context).status for _ in range(3)]

    assert statuses == ["failure", "failure", "max_retries_exceeded"]
    assert len(calls) == 2


def test_run_tests_without_allowed_dirs_skips_pytest(tmp_path, monkeypatch):
    calls = _fake_pytest(monkeypatch, 1)
    context = _context(tmp_path, trial=_trial())

    result = RunTestsStage().run(context)

    assert result.status == "success"
    assert "No test files found" in result.message
    assert calls == []
