"""Tests for espnet3.bin.atlas.stages.stage_status."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from espnet3.bin.atlas.core.models import Trial
from espnet3.bin.atlas.core.state import StateStore
from espnet3.bin.atlas.stages import stage_status

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


def test_stage_status_defaults_for_old_trials(tmp_path):
    assert stage_status._read_stage_status(tmp_path) == {"version": 1, "stages": {}}


def test_ensure_stage_statuses_adds_pending_stages_once(tmp_path):
    first = stage_status._ensure_stage_statuses(tmp_path, ["a", "b"])
    assert first["stages"]["a"]["status"] == "pending"
    assert first["stages"]["b"]["history"][0]["status"] == "pending"
    stamp = (tmp_path / "stage_status.yaml").read_text(encoding="utf-8")

    again = stage_status._ensure_stage_statuses(tmp_path, ["a", "b"])

    assert again["stages"]["a"] == first["stages"]["a"]
    assert (tmp_path / "stage_status.yaml").read_text(encoding="utf-8") == stamp


def test_set_stage_status_records_history_and_return_code(tmp_path):
    stage_status._ensure_stage_statuses(tmp_path, ["train"])
    stage_status._set_stage_status(tmp_path, "train", "running", "att1")
    stage_status._set_stage_status(tmp_path, "train", "failed", "att1", 3)
    stage_status._set_stage_status(tmp_path, "eval", "running", "att2")

    stages = stage_status._read_stage_status(tmp_path)["stages"]

    train = stages["train"]
    assert train["status"] == "failed" and train["return_code"] == 3
    assert [h["status"] for h in train["history"]] == ["pending", "running", "failed"]
    assert "return_code" not in train["history"][1]
    assert stages["eval"]["attempt_id"] == "att2"


def test_set_stage_status_caps_history_at_100_entries(tmp_path):
    for i in range(105):
        stage_status._set_stage_status(tmp_path, "s", "running", f"a{i}")
    history = stage_status._read_stage_status(tmp_path)["stages"]["s"]["history"]
    assert len(history) == 100
    assert history[-1]["attempt_id"] == "a104"


def test_invalidate_stages_resets_to_pending_and_keeps_history(tmp_path):
    stage_status._set_stage_status(tmp_path, "train", "success", "a1")

    stage_status._invalidate_stages(tmp_path, ["train", "new"], "config changed")

    stages = stage_status._read_stage_status(tmp_path)["stages"]
    assert stages["train"]["status"] == "pending"
    assert stages["train"]["previous_status"] == "success"
    assert stages["train"]["reason"] == "config changed"
    assert [h["status"] for h in stages["train"]["history"]] == ["success", "pending"]
    assert stages["new"]["previous_status"] is None
    # a later status change drops the stale previous_status marker
    stage_status._set_stage_status(tmp_path, "train", "running", "a2")
    after = stage_status._read_stage_status(tmp_path)["stages"]["train"]
    assert "previous_status" not in after


@pytest.mark.parametrize("text", ["- 1\n- 2\n", "stages: 3\n"])
def test_read_stage_status_rejects_malformed_files(tmp_path, text):
    (tmp_path / "stage_status.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid stage status"):
        stage_status._read_stage_status(tmp_path)
