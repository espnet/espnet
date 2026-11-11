"""CSV-backed Atlas state."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from espnet3.bin.atlas.core.models import JobRecord, StageResult, Study, Trial
from espnet3.bin.atlas.core.serialization import (
    atomic_write_text,
    dumps_json,
    file_lock,
    loads_json,
    utc_now,
)
from espnet3.bin.atlas.reports.csv_writer import sync_trials_csv

TABLES = ("studies", "trials", "node_runs", "jobs", "events")


class StateStore:
    """Persist state in one CSV file per record type.

    Each record type has a CSV file with one column per field; cells are JSON
    encoded to retain nulls, numbers, and nested values. A single advisory lock
    protects read/modify/write operations; failed non-blocking lock attempts
    retry every second. Rewrites use a temporary file and atomic rename.
    """

    def __init__(self, path: Path):
        path = Path(path)
        self.study_dir = path.parent if path.suffix == ".sqlite" else path
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.study_dir / ".state.lock"
        self._migrate_legacy(
            path if path.suffix == ".sqlite" else self.study_dir / "state.sqlite"
        )

    def _path(self, table: str) -> Path:
        return self.study_dir / f"state_{table}.csv"

    def _read_unlocked(self, table: str) -> list[dict]:
        path = self._path(table)
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as f:
            return [
                {key: loads_json(value) for key, value in row.items() if value != ""}
                for row in csv.DictReader(f)
            ]

    def _write_unlocked(self, table: str, rows: list[dict]) -> None:
        buffer = io.StringIO(newline="")
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: dumps_json(value, indent=None) for key, value in row.items()}
            )
        atomic_write_text(self._path(table), buffer.getvalue())

    def _rows(self, table: str) -> list[dict]:
        with file_lock(self.lock_path):
            return self._read_unlocked(table)

    def _upsert(self, table: str, key: str, value: dict) -> None:
        """Replace the record matching this table's primary key."""
        with file_lock(self.lock_path):
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

    def _migrate_legacy(self, db_path: Path) -> None:
        if not db_path.exists() or any(self._path(t).exists() for t in TABLES):
            return
        with sqlite3.connect(db_path) as conn, file_lock(self.lock_path):
            conn.row_factory = sqlite3.Row
            for table in TABLES:
                try:
                    rows = [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]
                except sqlite3.OperationalError:
                    rows = []
                if table == "events":
                    rows = [{"id": row.pop("id"), **row} for row in rows]
                self._write_unlocked(table, rows)

    def create_or_load_study(self, study_id: str, study_dir: Path) -> Study:
        with file_lock(self.lock_path):
            rows = self._read_unlocked("studies")
            for row in rows:
                if row["study_id"] == study_id:
                    row.pop("id", None)
                    return Study(**row)
            now = utc_now()
            study = Study(study_id, str(study_dir), "initialized", now, now)
            rows.append(asdict(study))
            self._write_unlocked("studies", rows)
            return study

    def get_study(self, study_id: str) -> Study:
        row = next(r for r in self._rows("studies") if r["study_id"] == study_id)
        row.pop("id", None)
        return Study(**row)

    def _update_study(self, study_id: str, **changes) -> None:
        with file_lock(self.lock_path):
            rows = self._read_unlocked("studies")
            for row in rows:
                if row["study_id"] == study_id:
                    row.update(changes, updated_at=utc_now())
                    break
            self._write_unlocked("studies", rows)

    def update_study_status(self, study_id: str, status: str) -> None:
        self._update_study(study_id, status=status)

    def get_agent_session_id(self, study_id: str) -> str | None:
        return self.get_study(study_id).agent_session_id

    def set_agent_session_id(self, study_id: str, session_id: str) -> None:
        self._update_study(study_id, agent_session_id=session_id)

    def agent_context_initialized(self, study_id: str) -> bool:
        return self.get_study(study_id).agent_context_initialized

    def mark_agent_context_initialized(self, study_id: str) -> None:
        self._update_study(study_id, agent_context_initialized=True)

    def set_best_trial(self, study_id: str, trial_id: str) -> None:
        self._update_study(study_id, current_best_trial_id=trial_id)

    @staticmethod
    def _row_to_trial(data: dict) -> Trial:
        return Trial(
            trial_id=data["trial_id"],
            study_id=data["study_id"],
            status=data["status"],
            config_patch=loads_json(data.get("config_patch_json") or "{}"),
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
            metrics=loads_json(data.get("metrics_json") or "{}"),
            reflection=data.get("reflection") or "",
            next_suggestion=data.get("next_suggestion") or "",
            codex_thread_id=data.get("codex_thread_id"),
            extra_stages=loads_json(data.get("extra_stages_json") or "[]"),
        )

    @staticmethod
    def _trial_row(trial: Trial) -> dict:
        return {
            "trial_id": trial.trial_id,
            "study_id": trial.study_id,
            "status": trial.status,
            "parent_trial_id": trial.parent_trial_id,
            "config_patch_json": dumps_json(trial.config_patch, None),
            "resolved_config_path": (
                str(trial.resolved_config_path) if trial.resolved_config_path else None
            ),
            "rationale": trial.rationale,
            "expected_effect": trial.expected_effect,
            "risk": trial.risk,
            "metrics_json": dumps_json(trial.metrics, None),
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
            "extra_stages_json": dumps_json(trial.extra_stages, None),
        }

    def create_trial(self, trial: Trial) -> Trial:
        self._upsert("trials", trial.trial_id, self._trial_row(trial))
        sync_trials_csv(self, self.study_dir, trial.study_id)
        return trial

    def update_trial(self, trial: Trial) -> None:
        trial.updated_at = utc_now()
        self._upsert("trials", trial.trial_id, self._trial_row(trial))
        sync_trials_csv(self, self.study_dir, trial.study_id)

    def get_trial(self, trial_id: str) -> Trial:
        return self._row_to_trial(
            next(r for r in self._rows("trials") if r["trial_id"] == trial_id)
        )

    def list_trials(self, study_id: str, status: str | None = None) -> list[Trial]:
        rows = [
            r
            for r in self._rows("trials")
            if r["study_id"] == study_id and (status is None or r["status"] == status)
        ]
        return [
            self._row_to_trial(r)
            for r in sorted(rows, key=lambda r: (r["created_at"], r["trial_id"]))
        ]

    def get_best_trial(self, study_id: str, metric: str, mode: str) -> Trial | None:
        trials = [
            t
            for t in self.list_trials(study_id)
            if t.score is not None and t.score_name == metric
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
        with file_lock(self.lock_path):
            rows = self._read_unlocked("node_runs")
            for row in rows:
                if row["run_id"] == run_id:
                    row.update(changes)
                    break
            self._write_unlocked("node_runs", rows)

    def begin_node_run(self, run_id: str, pid: int | None = None) -> None:
        self._update_node_run(
            run_id, status="running", started_at=utc_now(), local_pid=pid
        )

    def finish_node_run(self, run_id: str, result: StageResult) -> None:
        status = (
            "failed"
            if result.status in {"failure", "timeout", "retry"}
            else "completed"
        )
        self._update_node_run(
            run_id,
            status=status,
            finished_at=utc_now(),
            result_status=result.status,
            result_payload_json=result.to_json(),
        )

    def get_pending_node_runs(self, study_id: str) -> list[dict]:
        return sorted(
            [
                r
                for r in self._rows("node_runs")
                if r["study_id"] == study_id and r["status"] == "pending"
            ],
            key=lambda r: r["run_id"],
        )

    def get_node_run(self, run_id: str) -> dict | None:
        return next((r for r in self._rows("node_runs") if r["run_id"] == run_id), None)

    def list_node_runs(self, study_id: str) -> list[dict]:
        return sorted(
            [r for r in self._rows("node_runs") if r["study_id"] == study_id],
            key=lambda r: r["run_id"],
        )

    def create_job(self, job: JobRecord) -> None:
        self._upsert("jobs", job.job_id, asdict(job))

    def update_job_status(
        self, job_id: str, status: str, external_id: str | None = None
    ) -> None:
        with file_lock(self.lock_path):
            rows = self._read_unlocked("jobs")
            for row in rows:
                if row["job_id"] == job_id:
                    row["status"] = status
                    row["external_id"] = external_id or row.get("external_id")
                    row["updated_at"] = utc_now()
            self._write_unlocked("jobs", rows)

    def list_active_jobs(self, study_id: str) -> list[JobRecord]:
        active = {"submitted", "pending", "running", "unknown"}
        return [
            JobRecord(**{k: v for k, v in row.items() if k != "id"})
            for row in self._rows("jobs")
            if row["study_id"] == study_id and row["status"] in active
        ]

    def list_jobs(
        self, study_id: str, job_type: str | None = None
    ) -> list[JobRecord]:
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
            for row in self._rows("jobs")
            if row["study_id"] == study_id
            and (job_type is None or row["job_type"] == job_type)
        ]
        return [
            JobRecord(**{key: value for key, value in row.items() if key != "id"})
            for row in sorted(rows, key=lambda row: row["created_at"])
        ]

    def compute_trial_gpu_hours(self, study_id: str) -> float:
        now = datetime.now(timezone.utc)
        total = 0.0
        for row in self._rows("node_runs"):
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
        rows = [r for r in self._rows("jobs") if r.get("node_run_id") == node_run_id]
        if not rows:
            return None
        row = max(rows, key=lambda r: r["created_at"])
        return JobRecord(**{k: v for k, v in row.items() if k != "id"})

    def append_event(self, study_id: str, event_type: str, payload: dict) -> None:
        with file_lock(self.lock_path):
            rows = self._read_unlocked("events")
            rows.append(
                {
                    "id": len(rows) + 1,
                    "study_id": study_id,
                    "timestamp": utc_now(),
                    "event_type": event_type,
                    "payload_json": dumps_json(payload, None),
                }
            )
            self._write_unlocked("events", rows)
