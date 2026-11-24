from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from espnet3.bin.atlas.agents.interface import AgentRequest, AgentResponse
from espnet3.bin.atlas.agents.session import StudySessionAgentClient
from espnet3.bin.atlas.core.errors import ConfigValidationError
from espnet3.bin.atlas.core.models import Trial
from espnet3.bin.atlas.core.scheduler import AtlasGraphRuntime
from espnet3.bin.atlas.core.state import StateStore


def test_parse_training_progress_uses_latest_populated_row(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "epoch,step,train_loss,val_loss\n" "0,10,1.0,\n" "0,20,,0.9\n" "1,,0.8,\n",
        encoding="utf-8",
    )

    assert AtlasGraphRuntime._parse_training_progress(metrics, ("epoch",)) == 1
    assert AtlasGraphRuntime._parse_training_progress(metrics, ("step",)) == 20


def test_load_checked_keypoints_migrates_epoch_only_format(tmp_path):
    keypoints = tmp_path / "keypoints_checked.json"
    keypoints.write_text(json.dumps([2, 4]), encoding="utf-8")

    assert AtlasGraphRuntime._load_checked_keypoints(keypoints) == {
        "epoch": [2, 4],
        "iteration": [],
        "elapsed_time": [],
    }


def test_elapsed_trial_seconds_uses_utc_start_time():
    started = (datetime.now(timezone.utc) - timedelta(seconds=75)).isoformat()
    elapsed = AtlasGraphRuntime._elapsed_trial_seconds(started)

    assert elapsed is not None
    assert 74 <= elapsed <= 76


def test_early_stopping_rejects_multiple_progress_triggers():
    runtime = object.__new__(AtlasGraphRuntime)
    runtime.config = SimpleNamespace(
        atlas=SimpleNamespace(
            early_stopping=SimpleNamespace(
                epoch_interval=None,
                iteration_interval=1000,
                elapsed_time_interval_sec=0,
            )
        )
    )

    with pytest.raises(ConfigValidationError, match="only one trigger"):
        runtime._validate_early_stopping_config()


def test_study_session_is_reused_by_a_new_agent_process(tmp_path):
    state = StateStore(tmp_path / "state.sqlite")
    state.create_or_load_study("study", tmp_path)
    requests = []

    class FirstAgent:
        def run(self, request, artifact_dir=None):
            requests.append(request)
            return AgentResponse(
                status="success", message="", content="", thread_id="session-created"
            )

    request = AgentRequest(
        task="propose",
        objective="",
        knowledge="",
        repo_context="",
        trial_history_csv="",
        latest_metrics={},
        context_files={},
        allowed_actions=[],
        output_schema={},
    )
    StudySessionAgentClient(FirstAgent(), state, "study", tmp_path, tmp_path).run(
        request
    )
    assert state.get_agent_session_id("study") == "session-created"

    class LaterAgent:
        def run(self, request, artifact_dir=None):
            requests.append(request)
            return AgentResponse(status="success", message="", content="")

    StudySessionAgentClient(LaterAgent(), state, "study", tmp_path, tmp_path).run(
        request
    )
    assert requests[-1].resume_thread_id == "session-created"


def test_gpu_self_controller_releases_its_slot_before_submitting_next_trial():
    runtime = object.__new__(AtlasGraphRuntime)
    runtime.study_id = "study"
    calls = []
    job = SimpleNamespace(job_id="job_1", status="running", external_id="123")
    runtime.state = SimpleNamespace(
        find_job_by_node_run=lambda run_id: job,
        update_job_status=lambda *args, **kwargs: calls.append(
            ("update", args, kwargs)
        ),
        append_event=lambda *args, **kwargs: calls.append(("event", args, kwargs)),
    )
    runtime.run_node = lambda **kwargs: calls.append(
        ("run_node", kwargs)
    ) or SimpleNamespace(status="success")
    runtime._submit_pending_trial_nodes = lambda: calls.append(("submit",))

    result = runtime.run_gpu_self_controller(
        node_name="run_trial", run_id="run_1", trial_id="trial_000001"
    )

    assert result.status == "success"
    assert calls[0] == (
        "run_node",
        {
            "node_name": "run_trial",
            "run_id": "run_1",
            "trial_id": "trial_000001",
            "attempt_id": None,
            "controller_mode": True,
            "gpu_self_controller": True,
        },
    )
    assert calls[1][0] == "update"
    assert calls[-1] == ("submit",)


def test_iteration_and_elapsed_keypoints_share_one_agent_check(tmp_path):
    trial_id = "trial_000001"
    trial_dir = tmp_path / "trials" / trial_id
    metrics = trial_dir / "exp" / "csv_logs" / "version_0" / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("epoch,step,train_loss\n0,10,1.0\n", encoding="utf-8")
    (trial_dir / "resolved_training_config.yaml").write_text(
        "trainer:\n  max_epochs: 20\n", encoding="utf-8"
    )
    trial = Trial(
        trial_id=trial_id,
        study_id="study",
        status="running",
        config_patch={},
        resolved_config_path=None,
        rationale="",
        expected_effect="",
        risk="",
        score=None,
        score_name=None,
        decision=None,
        parent_trial_id=None,
        attempt_count=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=(datetime.now(timezone.utc) - timedelta(seconds=3601)).isoformat(),
        codex_thread_id="thread",
    )
    runtime = object.__new__(AtlasGraphRuntime)
    runtime.study_dir = tmp_path
    runtime.config = SimpleNamespace(
        atlas=SimpleNamespace(
            early_stopping=SimpleNamespace(
                enabled=True,
                epoch_interval=0,
                iteration_interval=10,
                elapsed_time_interval_sec=3600,
            )
        )
    )
    runtime.state = SimpleNamespace(get_trial=lambda _: trial)
    runtime.logger = logging.getLogger(__name__)
    calls = []
    runtime._collect_keypoint_comparison = lambda _: "No comparison"
    runtime._call_keypoint_agent = lambda _trial, prompt, _dir: calls.append(
        prompt
    ) or {"should_stop": False}

    runtime._maybe_check_keypoints(trial_id)

    assert len(calls) == 1
    assert "iteration 10" in calls[0]
    assert "elapsed GPU runtime 3600s" in calls[0]
    assert json.loads((trial_dir / "keypoints_checked.json").read_text()) == {
        "elapsed_time": [3600],
        "epoch": [],
        "iteration": [10],
    }


def _tick_job(job_id, status):
    return SimpleNamespace(
        job_id=job_id,
        job_type="tick",
        status=status,
        external_id=f"ext_{job_id}",
        backend="pbs",
        command_json="{}",
        resources_json=None,
        workdir=None,
        stdout_path=None,
        stderr_path=None,
    )


def _tick_runtime(jobs, agent_jobs):
    updates, events = [], []
    runtime = object.__new__(AtlasGraphRuntime)
    runtime.study_id = "study"
    runtime.logger = logging.getLogger(__name__)
    runtime.state = SimpleNamespace(
        list_active_jobs=lambda _sid: jobs,
        update_job_status=lambda job_id, status, external_id=None: updates.append(
            (job_id, status)
        ),
        append_event=lambda sid, kind, payload: events.append((kind, payload)),
    )
    asked = []

    def action(operation, context):
        asked.append((operation, [j["job_id"] for j in context["jobs"]]))
        if agent_jobs is None:
            raise RuntimeError("agent down")
        return {"jobs": agent_jobs}

    runtime._submission_action = action
    return runtime, updates, events, asked


def test_reconcile_tick_jobs_asks_agent_and_records_dead_tick():
    jobs = [_tick_job("tick_dead", "running"), _tick_job("tick_self", "running")]
    runtime, updates, events, asked = _tick_runtime(
        jobs, [{"job_id": "tick_dead", "status": "failed"}]
    )

    runtime.reconcile_tick_jobs(exclude_job_id="tick_self")

    assert asked == [("status", ["tick_dead"])]
    assert updates == [("tick_dead", "failed")]
    assert events[0][0] == "tick_job_state_reconciled"
    assert events[0][1]["source"] == "agent"
    assert events[0][1]["previous_status"] == "running"


def test_reconcile_tick_jobs_keeps_record_on_unknown_or_agent_failure():
    jobs = [_tick_job("tick_a", "running")]
    runtime, updates, events, _ = _tick_runtime(
        jobs, [{"job_id": "tick_a", "status": "bogus"}]
    )
    runtime.reconcile_tick_jobs()
    assert updates == [] and events == []

    runtime, updates, events, _ = _tick_runtime(jobs, None)
    runtime.reconcile_tick_jobs()
    assert updates == [] and events == []
