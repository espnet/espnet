"""Tests for espnet3.bin.atlas.stages.select_best."""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from espnet3.bin.atlas.core.models import Trial
from espnet3.bin.atlas.core.state import StateStore
from espnet3.bin.atlas.stages.select_best import SelectBestStage

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


def test_select_best_writes_sorted_leaderboard_and_best_files(tmp_path):
    config_file = tmp_path / "best.yaml"
    config_file.write_text("a: 1\n", encoding="utf-8")
    context = _context(tmp_path, trial=_trial("trial_000001", score=0.5))
    context.state.create_trial(
        _trial(
            "trial_000002",
            score=0.2,
            config_patch={"lr": 0.1},
            resolved_config_path=config_file,
            decision="accepted",
        )
    )
    context.state.create_trial(_trial("trial_000003"))

    result = SelectBestStage().run(context)

    assert result.status == "success"
    with open(result.artifacts["leaderboard"], encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["rank"], r["trial_id"]) for r in rows] == [
        ("1", "trial_000002"),
        ("2", "trial_000001"),
    ]
    assert rows[0]["decision"] == "accepted" and rows[1]["decision"] == ""
    best = load_yaml(context.study_dir / "best_trial.yaml")
    assert best["trial_id"] == "trial_000002"
    assert best["config_patch"] == {"lr": 0.1}
    assert (context.study_dir / "best_config.yaml").read_text() == "a: 1\n"


def test_select_best_sorts_descending_for_max_mode(tmp_path):
    context = _context(
        tmp_path,
        trial=_trial("trial_000001", score=0.5, score_name="acc"),
        atlas={"metric": {"name": "acc", "mode": "max"}},
    )
    context.state.create_trial(_trial("trial_000002", score=0.9, score_name="acc"))

    result = SelectBestStage().run(context)

    with open(result.artifacts["leaderboard"], encoding="utf-8", newline="") as fh:
        ids = [r["trial_id"] for r in csv.DictReader(fh)]
    assert ids == ["trial_000002", "trial_000001"]
    assert not (context.study_dir / "best_config.yaml").exists()


def test_select_best_without_scores_writes_only_a_header(tmp_path):
    context = _context(tmp_path, trial=_trial())

    result = SelectBestStage().run(context)

    lines = Path(result.artifacts["leaderboard"]).read_text().splitlines()
    assert lines == ["rank,trial_id,score,score_name,status,decision"]
    assert not (context.study_dir / "best_trial.yaml").exists()
