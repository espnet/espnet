"""CSV-backed Atlas state."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from espnet3.bin.atlas.core.models import JobRecord, StageResult, Study, Trial
from espnet3.bin.atlas.core.reports import sync_trials_csv
from espnet3.bin.atlas.core.utils import get_utc_now, write_text_atomically

TABLES = ("studies", "trials", "node_runs", "jobs", "events")


class StateStore:
    """Persist state in one CSV file per record type.

    Each record type has a CSV file with one column per field; cells are JSON
    encoded to retain nulls, numbers, and nested values. A single advisory lock
    protects read/modify/write operations; failed non-blocking lock attempts
    retry every second. Rewrites use a temporary file and atomic rename.
    """

    def __init__(self, path: Path):
        """Open (or create) a study's CSV state store.

        Args:
            path: The study directory.
        """
        self.study_dir = Path(path)
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.study_dir / ".state.lock"

    def _get_table_path(self, table: str) -> Path:
        """Return ``study_dir/state_<table>.csv`` for one of the known tables."""
        return self.study_dir / f"state_{table}.csv"

    def _read_unlocked(self, table: str) -> list[dict]:
        """Read and JSON-decode every row of a table's CSV, without locking."""
        path = self._get_table_path(table)
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as f:
            return [
                {key: json.loads(value) for key, value in row.items() if value != ""}
                for row in csv.DictReader(f)
            ]

    def _write_unlocked(self, table: str, rows: list[dict]) -> None:
        """Atomically overwrite a table's CSV with JSON-encoded ``rows`` (no lock)."""
        buffer = io.StringIO(newline="")
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    for key, value in row.items()
                }
            )
        write_text_atomically(self._get_table_path(table), buffer.getvalue())

    def _read_table_rows(self, table: str) -> list[dict]:
        """Read a table's rows while holding the study's state lock."""
        with FileLock(self.lock_path):
            return self._read_unlocked(table)

    def _upsert(self, table: str, key: str, value: dict) -> None:
        """Replace the record matching this table's primary key."""
        with FileLock(self.lock_path):
            rows = self._read_unlocked(table)
            key_field = {
                "studies": "study_id",
                "trials": "trial_id",
                "node_runs": "run_id",
                "jobs": "job_id",
                "events": "id",
            }[table]
            rows = [r for r in rows if str(r.get(key_field)) != key]
            rows.append(dict(value))
            self._write_unlocked(table, rows)

    def create_or_load_study(self, study_id: str, study_dir: Path) -> Study:
        """Return the existing study record, or create one if none exists yet.

        Args:
            study_id: The study's name/key.
            study_dir: The study's directory, recorded only when creating.

        Returns:
            Study: The existing or newly created study record.

        Example:
            >>> study = store.create_or_load_study("s", Path("exp/atlas/s"))
            >>> study.status
            'initialized'
        """
        with FileLock(self.lock_path):
            rows = self._read_unlocked("studies")
            for row in rows:
                if row["study_id"] == study_id:
                    row.pop("id", None)
                    return Study(**row)
            now = get_utc_now()
            study = Study(study_id, str(study_dir), "initialized", now, now)
            rows.append(asdict(study))
            self._write_unlocked("studies", rows)
            return study

    def get_study(self, study_id: str) -> Study:
        """Return a study's current record.

        Args:
            study_id: The study's name/key.

        Returns:
            Study: The study's record.

        Raises:
            StopIteration: If no study with this id exists.

        Example:
            >>> store.get_study("my_study").status
            'running'
        """
        rows = self._read_table_rows("studies")
        row = next(r for r in rows if r["study_id"] == study_id)
        row.pop("id", None)
        return Study(**row)

    def _update_study(self, study_id: str, **changes) -> None:
        """Merge ``changes`` into a study's row and bump its ``updated_at``."""
        with FileLock(self.lock_path):
            rows = self._read_unlocked("studies")
            for row in rows:
                if row["study_id"] == study_id:
                    row.update(changes, updated_at=get_utc_now())
                    break
            self._write_unlocked("studies", rows)

    def update_study_status(self, study_id: str, status: str) -> None:
        """Set a study's lifecycle status.

        Args:
            study_id: The study's name/key.
            status: New status, e.g. ``"running"``, ``"stopped"``,
                ``"completed"``.

        Returns:
            None.

        Example:
            >>> store.update_study_status("my_study", "stopped")
        """
        self._update_study(study_id, status=status)

    def get_agent_session_id(self, study_id: str) -> str | None:
        """Return a study's shared coding-agent session/thread id, if any.

        Args:
            study_id: The study's name/key.

        Returns:
            str | None: The persisted session id, or ``None`` if the agent
            has not been called yet.

        Example:
            >>> store.get_agent_session_id("my_study")
            'codex-session-abc123'
        """
        return self.get_study(study_id).agent_session_id

    def set_agent_session_id(self, study_id: str, session_id: str) -> None:
        """Persist a study's shared coding-agent session/thread id.

        Args:
            study_id: The study's name/key.
            session_id: The session/thread id to persist.

        Returns:
            None.

        Example:
            >>> store.set_agent_session_id("my_study", "codex-session-abc123")
        """
        self._update_study(study_id, agent_session_id=session_id)

    def is_agent_context_initialized(self, study_id: str) -> bool:
        """Whether the agent has already been given this study's full context.

        Args:
            study_id: The study's name/key.

        Returns:
            bool: ``True`` once a successful response has set this flag.

        Example:
            >>> store.is_agent_context_initialized("my_study")
            False
        """
        return self.get_study(study_id).is_agent_context_initialized

    def mark_agent_context_initialized(self, study_id: str) -> None:
        """Record that the agent has now seen this study's full context.

        Args:
            study_id: The study's name/key.

        Returns:
            None.

        Example:
            >>> store.mark_agent_context_initialized("my_study")
        """
        self._update_study(study_id, is_agent_context_initialized=True)

    def set_best_trial(self, study_id: str, trial_id: str) -> None:
        """Record a study's current best trial id.

        Args:
            study_id: The study's name/key.
            trial_id: The trial id to record as current best.

        Returns:
            None.

        Example:
            >>> store.set_best_trial("my_study", "trial_003")
        """
        self._update_study(study_id, current_best_trial_id=trial_id)

    @staticmethod
    def _convert_row_to_trial(data: dict) -> Trial:
        """Decode one ``trials`` CSV row (with JSON sub-fields) into a ``Trial``."""
        return Trial(
            trial_id=data["trial_id"],
            study_id=data["study_id"],
            status=data["status"],
            config_patch=json.loads(data.get("config_patch_json") or "{}"),
            resolved_config_path=(
                Path(data["resolved_config_path"])
                if data.get("resolved_config_path")
                else None
            ),
            rationale=data.get("rationale") or "",
            expected_effect=data.get("expected_effect") or "",
            risk=data.get("risk") or "",
            score=data.get("score"),
            score_name=data.get("score_name"),
            decision=data.get("decision"),
            parent_trial_id=data.get("parent_trial_id"),
            attempt_count=int(data.get("attempt_count") or 0),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            recovery_active=bool(data.get("recovery_active", False)),
            metrics=json.loads(data.get("metrics_json") or "{}"),
            reflection=data.get("reflection") or "",
            next_suggestion=data.get("next_suggestion") or "",
            codex_thread_id=data.get("codex_thread_id"),
            extra_stages=json.loads(data.get("extra_stages_json") or "[]"),
        )

    @staticmethod
    def _convert_trial_to_row(trial: Trial) -> dict:
        """Encode a ``Trial`` into a ``trials`` CSV row dict (with JSON sub-fields)."""
        return {
            "trial_id": trial.trial_id,
            "study_id": trial.study_id,
            "status": trial.status,
            "parent_trial_id": trial.parent_trial_id,
            "config_patch_json": json.dumps(trial.config_patch, ensure_ascii=False),
            "resolved_config_path": (
                str(trial.resolved_config_path) if trial.resolved_config_path else None
            ),
            "rationale": trial.rationale,
            "expected_effect": trial.expected_effect,
            "risk": trial.risk,
            "metrics_json": json.dumps(trial.metrics, ensure_ascii=False),
            "score": trial.score,
            "score_name": trial.score_name,
            "decision": trial.decision,
            "reflection": trial.reflection,
            "next_suggestion": trial.next_suggestion,
            "attempt_count": trial.attempt_count,
            "recovery_active": trial.recovery_active,
            "created_at": trial.created_at,
            "updated_at": trial.updated_at,
            "codex_thread_id": trial.codex_thread_id,
            "extra_stages_json": json.dumps(trial.extra_stages, ensure_ascii=False),
        }

    def create_trial(self, trial: Trial) -> Trial:
        """Persist a new trial and refresh the study's ``trials.csv`` report.

        Args:
            trial: The trial record to store.

        Returns:
            Trial: The same trial, unchanged (returned for chaining).

        Example:
            >>> store.create_trial(trial).trial_id
            'trial_001'
        """
        self._upsert("trials", trial.trial_id, self._convert_trial_to_row(trial))
        sync_trials_csv(self, self.study_dir, trial.study_id)
        return trial

    def update_trial(self, trial: Trial) -> None:
        """Persist a trial's current field values, bumping its ``updated_at``.

        Args:
            trial: The trial record to store (mutated in place: its
                ``updated_at`` is set to now before saving).

        Returns:
            None.

        Example:
            >>> trial.status = "accepted"
            >>> store.update_trial(trial)
        """
        trial.updated_at = get_utc_now()
        self._upsert("trials", trial.trial_id, self._convert_trial_to_row(trial))
        sync_trials_csv(self, self.study_dir, trial.study_id)

    def get_trial(self, trial_id: str) -> Trial:
        """Return one trial by id.

        Args:
            trial_id: The trial's id.

        Returns:
            Trial: The matching trial record.

        Raises:
            StopIteration: If no trial with this id exists.

        Example:
            >>> store.get_trial("trial_001").status
            'running'
        """
        return self._convert_row_to_trial(
            next(
                r for r in self._read_table_rows("trials") if r["trial_id"] == trial_id
            )
        )

    def list_trials(self, study_id: str, status: str | None = None) -> list[Trial]:
        """List a study's trials, ordered by creation time then trial id.

        Args:
            study_id: Study whose trials to list.
            status: Optional status to filter by (e.g. ``"running"``), or
                ``None`` for all trials.

        Returns:
            list[Trial]: Matching trials.

        Example:
            >>> len(store.list_trials("my_study", status="running"))
            1
        """
        rows = [
            r
            for r in self._read_table_rows("trials")
            if r["study_id"] == study_id and (status is None or r["status"] == status)
        ]
        return [
            self._convert_row_to_trial(r)
            for r in sorted(rows, key=lambda r: (r["created_at"], r["trial_id"]))
        ]

    def get_best_trial(
        self,
        study_id: str,
        metric: str,
        mode: str,
        exclude_trial_id: str | None = None,
    ) -> Trial | None:
        """Return the trial with the best score for a given metric.

        Args:
            study_id: Study whose trials to search.
            metric: Score-name to compare (only trials whose ``score_name``
                matches are considered).
            mode: ``"max"`` to pick the highest score, anything else to pick
                the lowest.
            exclude_trial_id: Trial to leave out, for example to ask what the
                best was before that trial.

        Returns:
            Trial | None: The best-scoring matching trial, or ``None`` if no
            trial has a score for this metric yet.

        Example:
            >>> store.get_best_trial("my_study", "wer", "min").trial_id
            'trial_002'
        """
        trials = [
            t
            for t in self.list_trials(study_id)
            if t.score is not None
            and t.score_name == metric
            and t.trial_id != exclude_trial_id
        ]
        if not trials:
            return None
        return sorted(trials, key=lambda t: float(t.score), reverse=mode == "max")[0]

    def create_node_run(
        self,
        run_id: str,
        study_id: str,
        node_name: str,
        trial_id: str | None = None,
        attempt: int = 0,
    ) -> None:
        """Create a new node run record with status ``"pending"``.

        Args:
            run_id: Unique id for this node run.
            study_id: Study the node run belongs to.
            node_name: Graph node to run.
            trial_id: Trial the node run operates on, or ``None`` for a
                study-level node.
            attempt: Attempt number, for a retried ``run_trial`` node run.

        Returns:
            None.

        Example:
            >>> store.create_node_run("run_1", "my_study", "research_context")
        """
        self._upsert(
            "node_runs",
            run_id,
            {
                "run_id": run_id,
                "study_id": study_id,
                "node_name": node_name,
                "trial_id": trial_id,
                "status": "pending",
                "attempt": attempt,
                "started_at": None,
                "finished_at": None,
                "result_status": None,
                "result_payload_json": None,
                "slurm_job_id": None,
                "local_pid": None,
            },
        )

    def _update_node_run(self, run_id: str, **changes) -> None:
        """Merge ``changes`` into one node run's row, by ``run_id``."""
        with FileLock(self.lock_path):
            rows = self._read_unlocked("node_runs")
            for row in rows:
                if row["run_id"] == run_id:
                    row.update(changes)
                    break
            self._write_unlocked("node_runs", rows)

    def begin_node_run(self, run_id: str, pid: int | None = None) -> None:
        """Mark a node run ``"running"`` and record its start time/local pid.

        Args:
            run_id: The node run to mark as started.
            pid: Local process id executing the node, or ``None`` if not
                running as a tracked local subprocess.

        Returns:
            None.

        Example:
            >>> store.begin_node_run("run_1")
        """
        self._update_node_run(
            run_id, status="running", started_at=get_utc_now(), local_pid=pid
        )

    def finish_node_run(self, run_id: str, result: StageResult) -> None:
        """Mark a node run ``"completed"`` or ``"failed"`` and record its result.

        Args:
            run_id: The node run to mark as finished.
            result: The stage's result; maps ``status`` in
                ``{"failure", "timeout", "retry"}`` to node-run status
                ``"failed"``, anything else to ``"completed"``.

        Returns:
            None.

        Example:
            >>> store.finish_node_run("run_1", StageResult(status="success"))
        """
        status = (
            "failed"
            if result.status in {"failure", "timeout", "retry"}
            else "completed"
        )
        self._update_node_run(
            run_id,
            status=status,
            finished_at=get_utc_now(),
            result_status=result.status,
            result_payload_json=result.to_json(),
        )

    def get_pending_node_runs(self, study_id: str) -> list[dict]:
        """List a study's not-yet-started node runs, ordered by run id.

        Args:
            study_id: Study whose node runs to list.

        Returns:
            list[dict]: Pending node-run rows (raw dicts, not a dataclass).

        Example:
            >>> len(store.get_pending_node_runs("my_study"))
            1
        """
        return sorted(
            [
                r
                for r in self._read_table_rows("node_runs")
                if r["study_id"] == study_id and r["status"] == "pending"
            ],
            key=lambda r: r["run_id"],
        )

    def get_node_run(self, run_id: str) -> dict | None:
        """Return one node run's row by id, or ``None`` if it does not exist.

        Args:
            run_id: The node run's id.

        Returns:
            dict | None: The matching node-run row, or ``None``.

        Example:
            >>> store.get_node_run("run_1")["node_name"]
            'research_context'
        """
        rows = self._read_table_rows("node_runs")
        return next((r for r in rows if r["run_id"] == run_id), None)

    def list_node_runs(self, study_id: str) -> list[dict]:
        """List all of a study's node runs, ordered by run id.

        Args:
            study_id: Study whose node runs to list.

        Returns:
            list[dict]: All node-run rows for the study.

        Example:
            >>> len(store.list_node_runs("my_study"))
            3
        """
        return sorted(
            [
                r
                for r in self._read_table_rows("node_runs")
                if r["study_id"] == study_id
            ],
            key=lambda r: r["run_id"],
        )

    def create_job(self, job: JobRecord) -> None:
        """Persist a new job record.

        Args:
            job: The job record to store.

        Returns:
            None.

        Example:
            >>> store.create_job(job_record)
        """
        self._upsert("jobs", job.job_id, asdict(job))

    def update_job_status(
        self, job_id: str, status: str, external_id: str | None = None
    ) -> None:
        """Update a job's status (and, once known, its backend-native id).

        Args:
            job_id: The job to update.
            status: New status, e.g. ``"running"``, ``"completed"``,
                ``"failed"``.
            external_id: Backend-native job id to set, or ``None`` to leave
                the existing value unchanged.

        Returns:
            None.

        Example:
            >>> store.update_job_status("job_1", "completed")
        """
        with FileLock(self.lock_path):
            rows = self._read_unlocked("jobs")
            for row in rows:
                if row["job_id"] == job_id:
                    row["status"] = status
                    row["external_id"] = external_id or row.get("external_id")
                    row["updated_at"] = get_utc_now()
            self._write_unlocked("jobs", rows)

    def list_active_jobs(self, study_id: str) -> list[JobRecord]:
        """List a study's jobs still in a non-terminal status.

        Args:
            study_id: Study whose jobs to list.

        Returns:
            list[JobRecord]: Jobs with status in ``{"submitted", "pending",
            "running", "unknown"}``.

        Example:
            >>> len(store.list_active_jobs("my_study"))
            1
        """
        active = {"submitted", "pending", "running", "unknown"}
        return [
            JobRecord(**{k: v for k, v in row.items() if k != "id"})
            for row in self._read_table_rows("jobs")
            if row["study_id"] == study_id and row["status"] in active
        ]

    def list_jobs(self, study_id: str, job_type: str | None = None) -> list[JobRecord]:
        """List study jobs, optionally filtered by type.

        Args:
            study_id: Study whose jobs should be returned.
            job_type: Optional job type such as ``tick`` or ``trial``.

        Returns:
            list[JobRecord]: Matching jobs ordered by creation time.

        Example:
            >>> store.list_jobs("study-1", job_type="tick")
            []
        """
        rows = [
            row
            for row in self._read_table_rows("jobs")
            if row["study_id"] == study_id
            and (job_type is None or row["job_type"] == job_type)
        ]
        return [
            JobRecord(**{key: value for key, value in row.items() if key != "id"})
            for row in sorted(rows, key=lambda row: row["created_at"])
        ]

    def compute_trial_gpu_hours(self, study_id: str) -> float:
        """Sum wall-clock GPU hours used across all of a study's trial node runs.

        For each trial-scoped node run with a recorded start time, adds the
        elapsed time from ``started_at`` to ``finished_at`` (or now, if still
        running).

        Args:
            study_id: Study to compute GPU hours for.

        Returns:
            float: Total elapsed hours across all trial node runs.

        Example:
            >>> store.compute_trial_gpu_hours("my_study")
            12.5
        """
        now = datetime.now(timezone.utc)
        total = 0.0
        for row in self._read_table_rows("node_runs"):
            if (
                row["study_id"] != study_id
                or row.get("trial_id") is None
                or not row.get("started_at")
            ):
                continue
            start = datetime.fromisoformat(row["started_at"])
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            end = (
                datetime.fromisoformat(row["finished_at"])
                if row.get("finished_at")
                else now
            )
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            total += max(0.0, (end - start).total_seconds() / 3600)
        return total

    def find_job_by_node_run(self, node_run_id: str) -> JobRecord | None:
        """Return the most recently created job submitted for a node run, if any.

        Args:
            node_run_id: The node run to look up.

        Returns:
            JobRecord | None: The latest matching job, or ``None`` if no job
            has been submitted for this node run.

        Example:
            >>> store.find_job_by_node_run("run_1").status
            'running'
        """
        rows = [
            r
            for r in self._read_table_rows("jobs")
            if r.get("node_run_id") == node_run_id
        ]
        if not rows:
            return None
        row = max(rows, key=lambda r: r["created_at"])
        return JobRecord(**{k: v for k, v in row.items() if k != "id"})

    def append_event(self, study_id: str, event_type: str, payload: dict) -> None:
        """Append one timestamped event row to the study's event log.

        Args:
            study_id: Study the event belongs to.
            event_type: Short event name, e.g. ``"node_started"``,
                ``"study_completed"``.
            payload: Event-specific JSON-serializable detail.

        Returns:
            None.

        Example:
            >>> store.append_event("my_study", "node_started", {"run_id": "run_1"})
        """
        with FileLock(self.lock_path):
            rows = self._read_unlocked("events")
            rows.append(
                {
                    "id": len(rows) + 1,
                    "study_id": study_id,
                    "timestamp": get_utc_now(),
                    "event_type": event_type,
                    "payload_json": json.dumps(
                        payload, ensure_ascii=False, default=str
                    ),
                }
            )
            self._write_unlocked("events", rows)
