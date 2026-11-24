from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone

from espnet3.bin.atlas.core.models import JobRecord, Trial
from espnet3.bin.atlas.core.state import StateStore


def _trial(study_id="study", trial_id="trial_000001"):
    """Build a minimal trial fixture with a unique timestamp."""
    now = datetime.now(timezone.utc).isoformat()
    return Trial(
        trial_id=trial_id,
        study_id=study_id,
        status="proposed",
        config_patch={"model.name": 'テスト, "quoted"', "values": [1, None]},
        resolved_config_path=None,
        rationale="測定",
        expected_effect="lower loss",
        risk="none",
        score=None,
        score_name="wer",
        decision=None,
        parent_trial_id=None,
        attempt_count=0,
        created_at=now,
        updated_at=now,
    )


def test_csv_state_round_trips_study_trial_and_nested_values(tmp_path):
    """Preserve typed and nested values across closing and reopening state."""
    store = StateStore(tmp_path)
    study = store.create_or_load_study("study", tmp_path)
    store.set_agent_session_id("study", "thread-1")
    store.mark_agent_context_initialized("study")
    store.set_best_trial("study", "trial_000001")
    trial = _trial()
    store.create_trial(trial)

    reopened = StateStore(tmp_path)
    assert reopened.get_study("study").study_id == study.study_id
    assert reopened.get_study("study").agent_session_id == "thread-1"
    assert reopened.is_agent_context_initialized("study") is True
    assert reopened.get_study("study").current_best_trial_id == "trial_000001"
    loaded = reopened.get_trial("trial_000001")
    assert loaded.config_patch == trial.config_patch
    assert loaded.rationale == "測定"
    assert loaded.score is None
    assert (tmp_path / "state_trials.csv").is_file()


def test_csv_state_updates_trial_and_filters_by_status(tmp_path):
    """Replace an existing trial row and query the updated state."""
    store = StateStore(tmp_path)
    store.create_or_load_study("study", tmp_path)
    trial = _trial()
    store.create_trial(trial)
    trial.status = "completed"
    trial.score = 0.25
    trial.metrics = {"dev": {"wer": 0.25}}
    store.update_trial(trial)

    assert store.get_trial(trial.trial_id).status == "completed"
    assert store.list_trials("study", status="proposed") == []
    assert store.list_trials("study", status="completed")[0].metrics == trial.metrics
    assert store.get_best_trial("study", "wer", "min").trial_id == trial.trial_id
    assert store.get_best_trial("study", "wer", "max").score == 0.25


def test_csv_state_tracks_active_jobs_and_node_runs(tmp_path):
    """Persist node execution and job transitions in their CSV tables."""
    store = StateStore(tmp_path)
    store.create_or_load_study("study", tmp_path)
    store.create_node_run("run-1", "study", "run_trial", "trial_000001")
    store.begin_node_run("run-1", pid=321)
    job = JobRecord(
        job_id="job-1",
        study_id="study",
        trial_id="trial_000001",
        node_run_id="run-1",
        backend="local",
        external_id="321",
        status="running",
        command_json='["python", "train"]',
        resources_json=None,
        workdir=str(tmp_path),
        stdout_path=None,
        stderr_path=None,
        job_type="trial",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store.create_job(job)

    assert store.get_node_run("run-1")["local_pid"] == 321
    assert store.get_pending_node_runs("study") == []
    assert store.list_node_runs("study")[0]["status"] == "running"
    assert store.list_active_jobs("study")[0].job_id == "job-1"
    assert store.find_job_by_node_run("run-1").external_id == "321"

    store.update_job_status("job-1", "completed")
    assert store.list_active_jobs("study") == []


def test_gpu_hours_include_completed_and_inflight_runs(tmp_path):
    """Count elapsed GPU time for finished and currently running trials."""
    store = StateStore(tmp_path)
    store.create_or_load_study("study", tmp_path)
    start = datetime.now(timezone.utc) - timedelta(hours=2)
    store.create_node_run("run-a", "study", "run_trial", "trial-a")
    store.begin_node_run("run-a")
    store._update_node_run(
        "run-a",
        started_at=(start - timedelta(hours=1)).isoformat(),
        finished_at=start.isoformat(),
    )
    store.create_node_run("run-b", "study", "run_trial", "trial-b")
    store.begin_node_run("run-b")
    # Backdate the in-flight run so the expected total is stable.
    store._update_node_run("run-b", started_at=start.isoformat())

    assert 2.99 <= store.compute_trial_gpu_hours("study") <= 3.01


def test_events_csv_preserves_json_payload(tmp_path):
    """Preserve nested event payload JSON through CSV quoting."""
    store = StateStore(tmp_path)
    store.create_or_load_study("study", tmp_path)
    store.append_event("study", "trial_created", {"message": 'a,b "c"', "count": 2})

    with (tmp_path / "state_events.csv").open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert json.loads(json.loads(row["payload_json"])) == {
        "message": 'a,b "c"',
        "count": 2,
    }


def test_cmd_status_reports_jobs_and_stop_file(tmp_path, capsys):
    """Summarise a study from recorded state without asking any scheduler."""
    import argparse

    from omegaconf import OmegaConf

    from espnet3.bin.atlas.command import handle_status

    OmegaConf.save(
        OmegaConf.create(
            {
                "atlas": {
                    "study_name": "study",
                    "metric": {"name": "wer", "mode": "min"},
                }
            }
        ),
        tmp_path / "atlas.yaml",
    )
    store = StateStore(tmp_path)
    store.create_or_load_study("study", tmp_path)
    trial = _trial()
    trial.status = "running"
    store.create_trial(trial)
    store.create_job(
        JobRecord(
            job_id="job-1",
            study_id="study",
            trial_id=trial.trial_id,
            node_run_id=None,
            backend="pbs",
            external_id="123.pbs",
            status="running",
            command_json="{}",
            resources_json=None,
            workdir=None,
            stdout_path=None,
            stderr_path=None,
            job_type="trial",
            created_at=trial.created_at,
            updated_at=trial.created_at,
        )
    )
    (tmp_path / "STOP").touch()

    assert handle_status(argparse.Namespace(study_dir=str(tmp_path))) == 0

    out = capsys.readouterr().out
    assert "trials: 1" in out
    assert "trial_status: running=1" in out
    assert "stop_file: yes" in out
    assert "active_jobs: 1" in out
    assert "trial running id=123.pbs" in out


def test_best_trial_can_leave_one_trial_out(tmp_path):
    """Ask for the best score as it was before a given trial."""
    store = StateStore(tmp_path)
    store.create_or_load_study("study", tmp_path)
    for trial_id, score in (("trial_000001", 0.5), ("trial_000002", 0.3)):
        trial = _trial(trial_id=trial_id)
        trial.score, trial.score_name = score, "wer"
        store.create_trial(trial)

    assert store.get_best_trial("study", "wer", "min").trial_id == "trial_000002"
    before = store.get_best_trial(
        "study", "wer", "min", exclude_trial_id="trial_000002"
    )
    assert before.trial_id == "trial_000001"
    assert (
        store.get_best_trial(
            "study", "wer", "min", exclude_trial_id="trial_000001"
        ).trial_id
        == "trial_000002"
    )
