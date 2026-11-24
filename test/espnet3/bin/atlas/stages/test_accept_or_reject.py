"""Tests for espnet3.bin.atlas.stages.accept_or_reject."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from espnet3.bin.atlas.core.models import Trial
from espnet3.bin.atlas.core.state import StateStore
from espnet3.bin.atlas.stages.accept_or_reject import AcceptOrRejectStage

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


def _decide(tmp_path, mode, old_score, new_score):
    """Run accept_or_reject for a new trial against an existing scored one."""
    context = _context(
        tmp_path,
        trial=_trial("trial_000002", score=new_score, status="completed"),
        atlas={"metric": {"name": "wer", "mode": mode}},
    )
    context.state.create_trial(_trial("trial_000001", score=old_score))
    return context, AcceptOrRejectStage().run(context)


@pytest.mark.parametrize(
    "mode,old,new,expected",
    [
        ("min", 0.5, 0.3, "accepted"),
        ("min", 0.5, 0.7, "rejected"),
        ("max", 0.5, 0.7, "accepted"),
        ("max", 0.5, 0.3, "rejected"),
    ],
)
def test_accept_or_reject_compares_scores_by_metric_mode(
    tmp_path, mode, old, new, expected
):
    context, result = _decide(tmp_path, mode, old, new)

    assert result.status == expected
    trial = context.state.get_trial("trial_000002")
    assert (trial.status, trial.decision) == (expected, expected)
    text = Path(result.artifacts["decision"]).read_text(encoding="utf-8")
    assert f"# Decision: {expected}" in text
    if expected == "accepted":
        best = context.state.get_study(STUDY).current_best_trial_id
        assert best == "trial_000002"
    assert f"best_before: {old}" in text


def test_accept_or_reject_accepts_the_first_scored_trial(tmp_path):
    context = _context(
        tmp_path,
        trial=_trial(score=0.9, status="completed"),
        atlas={"metric": {"name": "wer", "mode": "min"}},
    )

    result = AcceptOrRejectStage().run(context)

    assert result.status == "accepted"
    assert "best_before: n/a" in Path(result.artifacts["decision"]).read_text(
        encoding="utf-8"
    )


def test_accept_or_reject_accepts_when_no_trial_has_the_metric(tmp_path):
    context = _context(
        tmp_path,
        trial=_trial(score=0.9, score_name="cer", status="completed"),
        atlas={"metric": {"name": "wer", "mode": "min"}},
    )
    assert AcceptOrRejectStage().run(context).status == "accepted"
