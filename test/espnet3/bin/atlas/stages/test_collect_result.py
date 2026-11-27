"""Tests for espnet3.bin.atlas.stages.collect_result."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from espnet3.bin.atlas.core import AtlasError
from espnet3.bin.atlas.core.models import Trial
from espnet3.bin.atlas.core.state import StateStore
from espnet3.bin.atlas.stages.collect_result import (
    CollectResultStage,
    _extract_from_json,
    _extract_from_pattern,
    _extract_metrics,
    _resolve_primary_score,
)

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


def test_extract_from_json_resolves_literal_dotted_and_nested_keys(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps({"a.b": 1, "x": {"y": {"z": 2}}, "dev-clean.WER": {"k": 3}}),
        encoding="utf-8",
    )
    assert _extract_from_json(path, "a.b") == 1
    assert _extract_from_json(path, "x.y.z") == 2
    assert _extract_from_json(path, "dev-clean.WER.k") == 3


@pytest.mark.parametrize("key", ["nope", "x.nope", "x.y.z.deeper"])
def test_extract_from_json_raises_on_missing_key(tmp_path, key):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"x": {"y": {"z": 2}}}), encoding="utf-8")
    with pytest.raises(AtlasError, match="Missing JSON key"):
        _extract_from_json(path, key)


def test_extract_from_pattern_returns_the_first_number(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text("noise\nWER: 12.5 %\nWER: 9\n", encoding="utf-8")
    assert _extract_from_pattern(path, "WER: {value} %") == 12.5
    with pytest.raises(AtlasError, match="not found"):
        _extract_from_pattern(path, "CER={value}")


def test_extract_metrics_handles_json_mean_pattern_and_missing_files(tmp_path):
    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "scores.json").write_text(
        json.dumps(
            {
                "wer": 0.3,
                "sets": {"a": {"WER": 10, "CER": 1}, "b": {"WER": 20}, "c": 5},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "res" / "out.log").write_text("final loss=0.75\n", encoding="utf-8")
    rules = [
        {"path": "res/scores.json", "key": "wer", "name": "wer"},
        {"path": "res/*.json", "key": "sets", "aggregate": "mean", "name": "mean"},
        {
            "path": "res/*.json",
            "key": "sets",
            "aggregate": "mean",
            "sub_key": "CER",
            "name": "cer",
        },
        {"path": "res/scores.json", "key": "wer"},
        {"path": "res/out.log", "pattern": "loss={value}"},
        {"path": "res/out.log", "name": "ignored"},
        {"path": "missing/*.json", "key": "wer", "name": "absent"},
    ]
    metrics = _extract_metrics(tmp_path, rules)
    assert metrics == {"wer": 0.3, "mean": 15.0, "cer": 1.0, "out": 0.75}
    assert _extract_metrics(tmp_path, None) == {}


def test_extract_metrics_reads_attribute_style_rules(tmp_path):
    (tmp_path / "s.json").write_text('{"v": 4}', encoding="utf-8")
    rule = OmegaConf.create({"path": "s.json", "name": "v", "key": "v"})
    assert _extract_metrics(tmp_path, [rule]) == {"v": 4}


def test_extract_metrics_rejects_bad_mean_aggregation(tmp_path):
    (tmp_path / "s.json").write_text('{"v": 4, "d": {"a": {"x": 1}}}', encoding="utf-8")
    scalar = [{"path": "s.json", "key": "v", "aggregate": "mean", "name": "v"}]
    with pytest.raises(AtlasError, match="requires a dict"):
        _extract_metrics(tmp_path, scalar)
    empty = [{"path": "s.json", "key": "d", "aggregate": "mean", "name": "d"}]
    with pytest.raises(AtlasError, match="No 'WER' sub-values"):
        _extract_metrics(tmp_path, empty)


def test_resolve_primary_score_requires_the_named_metric():
    assert _resolve_primary_score({"wer": "0.5"}, "wer") == 0.5
    with pytest.raises(AtlasError, match="was not extracted"):
        _resolve_primary_score({}, "wer")


def _collect_atlas(rules):
    return {"metric": {"name": "wer", "mode": "min", "extraction": rules}}


def test_collect_result_stores_score_and_marks_trial_completed(tmp_path):
    context = _context(
        tmp_path,
        trial=_trial(),
        atlas=_collect_atlas([{"path": "m.json", "key": "wer", "name": "wer"}]),
    )
    (context.get_trial_dir() / "m.json").write_text('{"wer": 0.2}', encoding="utf-8")

    result = CollectResultStage().run(context)

    assert result.status == "success"
    assert result.payload == {"score": 0.2}
    stored = context.state.get_trial("trial_000001")
    assert (stored.score, stored.score_name, stored.status) == (0.2, "wer", "completed")
    assert stored.metrics == {"wer": 0.2}


def test_collect_result_keeps_an_accepted_status(tmp_path):
    context = _context(
        tmp_path,
        trial=_trial(status="accepted"),
        atlas=_collect_atlas([{"path": "m.json", "key": "wer", "name": "wer"}]),
    )
    (context.get_trial_dir() / "m.json").write_text('{"wer": 1}', encoding="utf-8")
    CollectResultStage().run(context)
    assert context.state.get_trial("trial_000001").status == "accepted"


def test_collect_result_fails_the_trial_when_the_score_is_missing(tmp_path):
    context = _context(tmp_path, trial=_trial(), atlas=_collect_atlas([]))

    result = CollectResultStage().run(context)

    assert result.status == "failure"
    assert "was not extracted" in result.message
    assert context.state.get_trial("trial_000001").status == "failed"
