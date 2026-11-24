"""Graph runtime and scheduler."""

from __future__ import annotations

import csv
import json
import logging
import os
import shlex
import sys
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock
from omegaconf import OmegaConf

from espnet3.bin.atlas.agents.session import StudySessionAgentClient
from espnet3.bin.atlas.core import AtlasError, InvalidAgentResponseError
from espnet3.bin.atlas.core.graph import Resources, load_stage_class
from espnet3.bin.atlas.core.models import AgentRequest, StageContext, StageResult
from espnet3.bin.atlas.core.state import JobRecord
from espnet3.bin.atlas.core.utils import (
    convert_pairs_to_dict,
    get_utc_now,
    write_text_atomically,
)
from espnet3.bin.atlas.resources import load_prompt, load_schema


class _SkillAgent:
    """Attach the graph-selected skill instructions to an agent request."""

    def __init__(self, agent, skill_path: Path):
        """Wrap ``agent`` so every request is given ``skill_path``'s instructions."""
        self.agent = agent
        self.skill_path = skill_path

    def run(self, request, artifact_dir=None):
        """Inject the skill's SKILL.md text and delegate the request to ``agent``."""
        context_files = dict(request.context_files or {})
        context_files["skill_instructions"] = self.skill_path.read_text(
            encoding="utf-8"
        )
        return self.agent.run(
            replace(request, context_files=context_files), artifact_dir=artifact_dir
        )


def _find_skill_path(recipe_dir: Path, skill: str) -> Path:
    """Find ``skill``'s ``SKILL.md`` under ``.agents/skills/`` near ``recipe_dir``."""
    if not skill or any(part in skill for part in ("/", "\\", "..")):
        raise ValueError(f"Invalid skill name: {skill!r}")
    for root in (recipe_dir, *recipe_dir.parents):
        path = root / ".agents" / "skills" / skill / "SKILL.md"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Skill '{skill}' not found under .agents/skills from {recipe_dir}"
    )


class AtlasGraphRuntime:
    """Drives one study's graph: runs/schedules nodes, submits jobs, tracks budget.

    Built once per ``init``/``run``/worker invocation via ``_load_runtime``
    and shared by all of a study's ``cmd_*`` commands and the standalone
    worker entry point.
    """

    def __init__(
        self,
        study_dir: Path,
        recipe_dir: Path,
        config,
        graph,
        state,
        trial_resources_default,
        agent,
    ) -> None:
        """Build the runtime from an already-loaded config/graph/state.

        Args:
            study_dir: Study's root directory.
            recipe_dir: Recipe root directory.
            config: The study's resolved ``atlas.yaml`` config.
            graph: The study's loaded ``ResearchGraph``.
            state: The study's ``StateStore``.
            trial_resources_default: Default ``Resources`` for trial jobs,
                merged per-node with each node's own ``resources``.
            agent: The raw coding-agent client (e.g. ``CliAgent``); wrapped
                here in a ``StudySessionAgentClient`` for session reuse.
        """
        self.study_dir = Path(study_dir).resolve()
        self.recipe_dir = Path(recipe_dir).resolve()
        self.config = config
        self.graph = graph
        self.state = state
        self.trial_resources_default = trial_resources_default
        self.study_id = str(config.atlas.study_name)
        self.agent = StudySessionAgentClient(
            agent,
            state,
            self.study_id,
            self.study_dir,
            self.recipe_dir,
            configured_session_id=getattr(config.atlas.agent, "session_id", None),
        )
        self.logger = logging.getLogger("espnet3.bin.atlas")
        self._validate_early_stopping_config()

    def _validate_early_stopping_config(self) -> None:
        """Require early stopping to use at most one progress clock."""
        early_stop_cfg = getattr(self.config.atlas, "early_stopping", None)
        if early_stop_cfg is None:
            return

        epoch_value = getattr(early_stop_cfg, "epoch_interval", None)
        try:
            epoch_enabled = epoch_value is None or int(epoch_value) > 0
            iteration_enabled = (
                int(getattr(early_stop_cfg, "iteration_interval", 0) or 0) > 0
            )
            elapsed_enabled = (
                int(getattr(early_stop_cfg, "elapsed_time_interval_sec", 0) or 0) > 0
            )
        except (TypeError, ValueError) as exc:
            raise AtlasError(
                "early_stopping intervals must be null or integer seconds/iterations."
            ) from exc

        enabled = [
            name
            for name, is_enabled in (
                ("epoch_interval", epoch_enabled),
                ("iteration_interval", iteration_enabled),
                ("elapsed_time_interval_sec", elapsed_enabled),
            )
            if is_enabled
        ]
        if len(enabled) > 1:
            raise AtlasError(
                "early_stopping accepts only one trigger, but multiple are enabled: "
                f"{', '.join(enabled)}. Set unused intervals to 0. "
                "In particular, set epoch_interval: 0 when using iteration_interval "
                "or elapsed_time_interval_sec."
            )

    def get_gpu_monitor_interval_sec(self) -> int:
        """Return how often a GPU job's self-controller polls its trial, in seconds.

        This is ``atlas.tick_interval_sec_trial_active`` (default 600), with a
        floor of 1 second.

        Returns:
            int: Poll interval in seconds, at least 1.

        Example:
            >>> runtime.get_gpu_monitor_interval_sec()
            600
        """
        configured = getattr(self.config.atlas, "tick_interval_sec_trial_active", 600)
        return max(1, int(configured or 600))

    def get_bootstrap_node_name(self) -> str:
        """Pick the first node to run for a study with no node runs yet.

        A study whose knowledge pack already exists skips the setup node and
        starts at ``propose_trial``; any other study starts at the graph's
        declared ``start`` node.

        Returns:
            str: The node name to create the first node run for.

        Example:
            >>> runtime.get_bootstrap_node_name()
            'research_context'
        """
        knowledge_pack = self.study_dir / "knowledge" / "knowledge_pack.md"
        if knowledge_pack.is_file() and "propose_trial" in self.graph.nodes:
            return "propose_trial"
        return self.graph.start

    def _build_job_name(
        self, node_name: str, trial_id: str | None, attempt_id: str
    ) -> str:
        """Build a short, backend-friendly job name from the study/trial/node."""
        study = self.study_dir.name.replace("_", "-")
        if trial_id:
            trial = trial_id
            if trial.startswith("trial_"):
                suffix = trial.split("_", 1)[1]
                if suffix.isdigit():
                    trial = f"trial-{int(suffix):04d}"
                else:
                    trial = trial.replace("_", "-")
            else:
                trial = trial.replace("_", "-")
            return f"ar-{study}-{trial}"[:200]
        node = node_name.replace("_", "-")
        return f"ar-{study}-{node}"[:200]

    def run_node(
        self,
        node_name: str,
        run_id: str,
        trial_id: str | None = None,
        attempt_id: str | None = None,
        controller_mode: bool = False,
        gpu_self_controller: bool = False,
    ) -> StageResult:
        """Execute one graph node's stage and, in controller mode, route to the next.

        Builds the node's ``StageContext`` (wrapping the agent with
        ``_SkillAgent`` if the node declares a ``skill``), records
        ``node_started``/``node_finished`` events, runs the stage (catching
        ``InvalidAgentResponseError`` into a failure result instead of
        propagating it), and — unless the result reports an invalid agent
        response — calls :meth:`_schedule_next` to advance the graph.

        Args:
            node_name: Graph node to run.
            run_id: This node run's unique id.
            trial_id: Trial the node operates on, or ``None`` for a
                study-level node.
            attempt_id: Attempt id within this node run; defaults to
                ``"attempt_000000"`` when ``None``.
            controller_mode: Whether this call is itself acting as the
                controller (so :meth:`_schedule_next` may recurse into the
                next local node immediately).
            gpu_self_controller: Whether this run is executing inside a GPU
                job acting as its own controller.

        Returns:
            StageResult: The stage's result.

        Example:
            >>> result = runtime.run_node("propose_trial", run_id="run_1")
            >>> result.status
            'success'
        """
        node_cfg = self.graph.nodes[node_name]
        stage_cls = load_stage_class(node_cfg.target, recipe_dir=self.recipe_dir)
        stage = stage_cls()
        trial = self.state.get_trial(trial_id) if trial_id else None
        if attempt_id is None:
            attempt_id = "attempt_000000"
        context = StageContext(
            study_dir=self.study_dir,
            recipe_dir=self.recipe_dir,
            node_name=node_name,
            run_id=run_id,
            attempt_id=attempt_id,
            graph=self.graph,
            config=self.config,
            state=self.state,
            agent=(
                _SkillAgent(
                    self.agent, _find_skill_path(self.recipe_dir, node_cfg.skill)
                )
                if node_cfg.skill
                else self.agent
            ),
            logger=self.logger,
            scheduler=self,
            current_trial=trial,
            gpu_self_controller=gpu_self_controller,
        )
        self.state.begin_node_run(run_id)
        self.state.append_event(
            self.study_id,
            "node_started",
            {"run_id": run_id, "node_name": node_name, "trial_id": trial_id},
        )
        try:
            result = stage.execute(context)
        except InvalidAgentResponseError as exc:
            result = StageResult(
                status="failure",
                message=str(exc),
                payload={"invalid_agent_response": True},
            )
            write_text_atomically(
                context.get_artifact_dir() / ".stage_failure.json", result.to_json()
            )
        self.state.finish_node_run(run_id, result)
        self.state.append_event(
            self.study_id,
            "node_finished",
            {
                "run_id": run_id,
                "node_name": node_name,
                "trial_id": trial_id,
                "status": result.status,
                "message": result.message,
            },
        )
        if result.payload.get("invalid_agent_response"):
            return result
        self._schedule_next(
            current_node=node_name,
            result=result,
            current_trial_id=trial_id,
            controller_mode=controller_mode,
        )
        return result

    def run_gpu_self_controller(
        self,
        node_name: str,
        run_id: str,
        trial_id: str | None = None,
        attempt_id: str | None = None,
    ) -> StageResult:
        """Finish a trial and advance its local graph nodes inside the GPU job.

        Runs ``node_name`` as usual, then — since in this mode the trial's
        own GPU job is acting as its controller rather than being observed
        externally — marks that job's own record ``completed``/``failed``
        itself, and submits any now-pending trial nodes.

        Args:
            node_name: Graph node to run (typically ``run_trial`` finishing).
            run_id: This node run's unique id.
            trial_id: Trial the node operates on, or ``None``.
            attempt_id: Attempt id within this node run, or ``None`` to
                default to ``"attempt_000000"``.

        Returns:
            StageResult: The stage's result, as from :meth:`run_node`.

        Example:
            >>> result = runtime.run_gpu_self_controller(
            ...     "run_trial", run_id="run_1", trial_id="trial_001"
            ... )
        """
        result = self.run_node(
            node_name=node_name,
            run_id=run_id,
            trial_id=trial_id,
            attempt_id=attempt_id,
            controller_mode=True,
            gpu_self_controller=True,
        )

        # A normal controller would observe this job through Slurm.  In this
        # mode the worker is the controller, so it must release its own slot
        # before selecting and submitting the next trial.
        job = self.state.find_job_by_node_run(run_id)
        if job and job.status in {"submitted", "pending", "running", "unknown"}:
            terminal_status = (
                "failed" if result.status in {"failure", "timeout"} else "completed"
            )
            self.state.update_job_status(
                job.job_id, terminal_status, external_id=job.external_id
            )
            self.state.append_event(
                self.study_id,
                "gpu_self_controller_job_finished",
                {
                    "job_id": job.job_id,
                    "run_id": run_id,
                    "status": terminal_status,
                },
            )
        self._submit_pending_trial_nodes()
        return result

    def _build_node_worker_command(
        self,
        node_name: str,
        run_id: str,
        trial_id: str | None,
        attempt_id: str,
    ) -> list[str]:
        """Build the ``espnet3.bin.atlas.worker`` argv for a submitted node run."""
        command = [
            sys.executable,
            "-m",
            "espnet3.bin.atlas.worker",
            "--study-dir",
            str(self.study_dir),
            "--recipe-dir",
            str(self.recipe_dir),
            "--node",
            node_name,
            "--run-id",
            run_id,
            "--attempt-id",
            attempt_id,
        ]
        if trial_id:
            command.extend(["--trial-id", trial_id])
        return command

    def _schedule_next(
        self,
        current_node: str,
        result: StageResult,
        current_trial_id: str | None,
        controller_mode: bool = False,
    ) -> None:
        """Route a finished node's result to its next node and create that node run.

        Updates ``recovery_active`` on a ``run_trial`` result, routes via
        ``result.next_node`` or ``graph.route()`` (treating a
        ``propose_trial``+``"duplicate"`` result as a dead end rather than an
        error), marks the study ``"completed"`` when there is no next node
        and no in-flight trial work remains (deferring completion
        otherwise), and — for a node routed to ``"local"`` execution while
        already in controller mode — runs it immediately rather than waiting
        for the next tick.
        """
        if (self.study_dir / "STOP").exists():
            return
        if current_node == "run_trial" and current_trial_id:
            trial = self.state.get_trial(current_trial_id)
            if result.status in {"failure", "timeout"}:
                trial.recovery_active = True
                self.state.update_trial(trial)
            elif result.status in {"success", "early_stopped"}:
                trial.recovery_active = False
                self.state.update_trial(trial)
        payload_trial_id = result.payload.get("trial_id")
        if isinstance(payload_trial_id, str) and payload_trial_id:
            current_trial_id = payload_trial_id
        next_node = (
            result.next_node
            if result.next_node
            else (
                None
                if current_node == "propose_trial" and result.status == "duplicate"
                else self.graph.route(current_node, result.status)
            )
        )
        if next_node is None:
            if self._has_inflight_trial_work():
                self.state.append_event(
                    self.study_id,
                    "study_completion_deferred",
                    {"current_node": current_node, "status": result.status},
                )
                return
            self.state.update_study_status(self.study_id, "completed")
            self.state.append_event(
                self.study_id,
                "study_completed",
                {"current_node": current_node, "status": result.status},
            )
            return
        # A proposal is study-scoped: it must not inherit the trial that
        # triggered reflection/update_knowledge.  Otherwise global ticks skip
        # the local node while the parallel-slot accounting still counts it.
        if next_node == "propose_trial":
            current_trial_id = None
        next_cfg = self.graph.nodes[next_node]
        next_run_id = f"run_{uuid.uuid4().hex}"
        attempt = 0
        attempt_id = "attempt_000000"
        if current_trial_id:
            trial = self.state.get_trial(current_trial_id)
            attempt = (
                trial.attempt_count + 1
                if next_node == "run_trial"
                else trial.attempt_count
            )
            attempt_id = f"attempt_{attempt:06d}"
        self.state.create_node_run(
            run_id=next_run_id,
            study_id=self.study_id,
            node_name=next_node,
            trial_id=current_trial_id,
            attempt=attempt,
        )
        self.state.append_event(
            self.study_id,
            "node_scheduled",
            {
                "run_id": next_run_id,
                "node_name": next_node,
                "trial_id": current_trial_id,
            },
        )
        if next_cfg.execution == "local" and controller_mode:
            self.run_node(
                node_name=next_node,
                run_id=next_run_id,
                trial_id=current_trial_id,
                attempt_id=attempt_id,
                controller_mode=True,
            )

    def should_stop(self) -> bool:
        """Return whether the controller should stop scheduling study work.

        Args:
            None: This method reads the configured study and its persisted state.

        Returns:
            bool: True when a stop request, completed study, or budget limit applies.

        Example:
            >>> runtime.should_stop()
            False
        """
        budget = self.config.atlas.budget
        stop_file = self.study_dir / "STOP"
        if stop_file.exists():
            return True
        if self.state.get_study(self.study_id).status == "completed":
            return True
        max_wallclock_hours = float(getattr(budget, "max_wallclock_hours", 0) or 0)
        if max_wallclock_hours > 0:
            study = self.state.get_study(self.study_id)
            created_at = datetime.fromisoformat(study.created_at)
            now = datetime.now(timezone.utc)
            elapsed_hours = (now - created_at).total_seconds() / 3600.0
            if elapsed_hours >= max_wallclock_hours:
                return True
        max_gpu_hours = float(getattr(budget, "max_gpu_hours", 0) or 0)
        if max_gpu_hours > 0:
            used_gpu_hours = self.state.compute_trial_gpu_hours(self.study_id)
            if used_gpu_hours >= max_gpu_hours:
                return True
        trials = self.state.list_trials(self.study_id)
        if len(trials) >= int(getattr(budget, "max_trials", 0) or 0) > 0:
            return True
        parent_ids_with_retry = {
            trial.parent_trial_id
            for trial in trials
            if trial.parent_trial_id is not None
        }
        failures = sum(
            1
            for trial in trials
            if trial.status in {"failed", "timeout", "cancelled"}
            and trial.trial_id not in parent_ids_with_retry
        )
        if failures >= int(getattr(budget, "max_failures", 0) or 0) > 0:
            return True
        no_improve = int(getattr(budget, "no_improve_stop", 0) or 0)
        if no_improve > 0:
            streak = 0
            for trial in reversed(trials):
                if trial.status not in {"rejected", "accepted"}:
                    continue
                if trial.status == "accepted":
                    break
                streak += 1
            if streak >= no_improve and trials:
                return True
        return False

    def _has_inflight_trial_work(self) -> bool:
        """Whether any active trial job or pending/running trial node run remains."""
        active_trial_jobs = any(
            job.job_type == "trial"
            for job in self.state.list_active_jobs(self.study_id)
        )
        if active_trial_jobs:
            return True
        return any(
            node_run["trial_id"] is not None
            and node_run["status"] in {"pending", "running"}
            for node_run in self.state.list_node_runs(self.study_id)
        )

    def _refresh_active_jobs(
        self,
        trial_id: str | None = None,
        include_trial_jobs: bool = True,
    ) -> None:
        """Ask the agent for current status of active (non-tick) jobs and reconcile.

        Queries the configured submission agent for every active job
        matching the given filters, updates each job's persisted status, and
        calls :meth:`_reconcile_terminal_job` for any that just reached a
        terminal state. Does nothing if there are no matching active jobs,
        or logs and returns if the status query itself fails.

        Args:
            trial_id: If given, restrict to jobs belonging to this trial;
                ``None`` checks all trials' jobs.
            include_trial_jobs: Whether to include ``job_type == "trial"``
                jobs in the query (set ``False`` to check only non-trial
                jobs, e.g. during a per-trial-only refresh).
        """
        jobs = [
            job
            for job in self.state.list_active_jobs(self.study_id)
            if job.job_type != "tick"
            and (trial_id is None or job.trial_id == trial_id)
            and (include_trial_jobs or job.job_type != "trial")
        ]
        if not jobs:
            return
        statuses = self._query_job_statuses(jobs)
        if statuses is None:
            return
        for job in jobs:
            status = statuses.get(job.job_id, "unknown")
            self.state.update_job_status(
                job.job_id, status, external_id=job.external_id
            )
            if status in {"completed", "failed", "cancelled", "timeout"}:
                self._reconcile_terminal_job(job, status)

    def _query_job_statuses(self, jobs) -> dict[str, str] | None:
        """Ask the submission agent for the current status of ``jobs``.

        Backend-agnostic: the agent follows ``submission.instructions`` to
        inspect each job with whatever scheduler the study uses.

        Args:
            jobs: ``JobRecord`` objects to inspect.

        Returns:
            dict[str, str] | None: ``job_id`` -> normalized status (one of
            ``submitted``, ``pending``, ``running``, ``completed``, ``failed``,
            ``cancelled``, ``timeout``, ``unknown``; anything else, or a job the
            agent did not report, becomes ``unknown``). ``None`` if the agent
            call failed.
        """
        payload = [
            {
                "job_id": job.job_id,
                "backend": job.backend,
                "external_id": job.external_id,
                "status": job.status,
                "job_type": job.job_type,
                "command": json.loads(job.command_json).get("command", []),
                "metadata": json.loads(job.command_json).get("metadata", {}),
                "resources": json.loads(job.resources_json or "{}"),
                "workdir": job.workdir,
                "stdout": job.stdout_path,
                "stderr": job.stderr_path,
            }
            for job in jobs
        ]
        try:
            result = self._run_submission_action("status", {"jobs": payload})
        except Exception as exc:
            self.logger.warning("Job status agent failed: %s", exc)
            return None
        reported = {
            str(item.get("job_id")): str(item.get("status", "unknown"))
            for item in result.get("jobs", [])
            if isinstance(item, dict)
        }
        valid_statuses = {
            "submitted",
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
            "timeout",
            "unknown",
        }
        return {
            job.job_id: (
                reported.get(job.job_id, "unknown")
                if reported.get(job.job_id, "unknown") in valid_statuses
                else "unknown"
            )
            for job in jobs
        }

    def reconcile_tick_jobs(self, exclude_job_id: str | None = None) -> None:
        """Repair recorded controller-tick job statuses via the submission agent.

        A tick job that died (crash, walltime, cancellation) can stay recorded
        as active, which would stop a replacement controller from being
        submitted. This asks the agent for the real state of every active tick
        job and updates the record. Unlike trial jobs, a tick job reaching a
        terminal state triggers no trial reconciliation. ``unknown`` answers and
        agent failures leave the record unchanged.

        Args:
            exclude_job_id: A tick job to skip, normally the caller's own job,
                which is running by definition.
        """
        ticks = [
            job
            for job in self.state.list_active_jobs(self.study_id)
            if job.job_type == "tick"
            and job.job_id != exclude_job_id
            and job.external_id
        ]
        if not ticks:
            return
        statuses = self._query_job_statuses(ticks)
        if statuses is None:
            return
        for job in ticks:
            status = statuses.get(job.job_id, "unknown")
            if status in {"unknown", job.status}:
                continue
            self.state.update_job_status(
                job.job_id, status, external_id=job.external_id
            )
            self.state.append_event(
                self.study_id,
                "tick_job_state_reconciled",
                {
                    "job_id": job.job_id,
                    "external_id": job.external_id,
                    "previous_status": job.status,
                    "status": status,
                    "source": "agent",
                },
            )

    def _run_submission_action(self, operation: str, context: dict) -> dict:
        """Ask the configured coding agent to submit or inspect jobs."""
        submission = self.config.atlas.submission
        instructions = str(getattr(submission, "instructions", "") or "")
        budget = self.config.atlas.budget
        max_wallclock = float(getattr(budget, "max_wallclock_hours", 0) or 0)
        study = self.state.get_study(self.study_id)
        created_at = datetime.fromisoformat(study.created_at)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        elapsed = max(
            0.0, (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
        )
        trials = self.state.list_trials(self.study_id)
        active_jobs = self.state.list_active_jobs(self.study_id)
        used_gpu_hours = self.state.compute_trial_gpu_hours(self.study_id)
        max_gpu_hours = float(getattr(budget, "max_gpu_hours", 0) or 0)
        failed_trials = sum(
            trial.status in {"failed", "timeout", "cancelled"} for trial in trials
        )
        max_failures = int(getattr(budget, "max_failures", 0) or 0)
        budget_context = {
            "limits": {
                key: getattr(budget, key, None)
                for key in (
                    "max_trials",
                    "max_parallel_trials",
                    "max_failures",
                    "no_improve_stop",
                    "max_wallclock_hours",
                    "max_gpu_hours",
                )
            },
            "trials_used": len(trials),
            "remaining_trials": (
                max(
                    0,
                    int(getattr(budget, "max_trials", 0) or 0) - len(trials),
                )
                if int(getattr(budget, "max_trials", 0) or 0) > 0
                else None
            ),
            "failed_trials": failed_trials,
            "remaining_failures": (
                max(0, max_failures - failed_trials) if max_failures > 0 else None
            ),
            "active_trial_jobs": sum(job.job_type == "trial" for job in active_jobs),
            "remaining_parallel_slots": max(
                0,
                int(getattr(budget, "max_parallel_trials", 1) or 1)
                - sum(job.job_type == "trial" for job in active_jobs),
            ),
            "elapsed_wallclock_hours": elapsed,
            "remaining_wallclock_hours": (
                max(0.0, max_wallclock - elapsed) if max_wallclock > 0 else None
            ),
            "used_gpu_hours": used_gpu_hours,
            "remaining_gpu_hours": (
                max(0.0, max_gpu_hours - used_gpu_hours) if max_gpu_hours > 0 else None
            ),
        }
        if max_wallclock > 0:
            budget_context["remaining_wallclock_hours"] = max(
                0.0, max_wallclock - elapsed
            )
        request_context = {
            "operation": operation,
            "instructions": instructions,
            "budget": budget_context,
            **context,
        }
        prompt = load_prompt("job_operate", instructions=instructions or "(none)")
        request_context["agent_task_instructions"] = prompt
        request = AgentRequest(
            task=f"job_{operation}",
            objective=prompt,
            knowledge="",
            repo_context="",
            trial_history_csv="",
            latest_metrics={},
            context_files={
                "submission_request": json.dumps(
                    request_context, ensure_ascii=False, indent=2, default=str
                )
            },
            allowed_actions=[
                "inspect the supplied study and job context",
                "submit or query only the supplied job(s)",
            ],
            output_schema=load_schema(f"job_{operation}"),
        )
        artifact_dir = (
            self.study_dir / "agent" / "submission" / f"{operation}_{uuid.uuid4().hex}"
        )
        response = self.agent.run(request, artifact_dir=artifact_dir)
        if response.status != "success":
            raise RuntimeError(
                f"Job {operation} agent failed: {response.message}: {response.content}"
            )
        if not response.structured:
            raise RuntimeError(
                f"Job {operation} agent returned invalid JSON: {response.content}"
            )
        return response.structured

    def submit_job(
        self,
        *,
        job_id: str,
        job_type: str,
        name: str,
        command: list[str],
        workdir: Path,
        stdout: Path,
        stderr: Path,
        resources: Resources,
        env: dict[str, str] | None = None,
        metadata: dict | None = None,
        trial_id: str | None = None,
        node_run_id: str | None = None,
    ) -> JobRecord:
        """Delegate one asynchronous job submission to the configured agent.

        Asks the configured submission agent (Slurm, local, or custom) to
        launch ``command`` with the given resources/environment, then
        persists a ``JobRecord`` for it in state.

        Args:
            job_id: Atlas-assigned id for this job.
            job_type: Kind of job, e.g. ``"trial"`` or ``"tick"``.
            name: Human-readable job name (e.g. shown in ``squeue``).
            command: Argv to run.
            workdir: Working directory for the job.
            stdout: Path to redirect stdout to.
            stderr: Path to redirect stderr to.
            resources: Resource limits to request.
            env: Extra environment variables for the job process, or
                ``None``.
            metadata: Extra metadata recorded alongside the job, or
                ``None``.
            trial_id: Trial this job belongs to, or ``None`` for a
                study-level job.
            node_run_id: Graph node run this job executes, or ``None``.

        Returns:
            JobRecord: The persisted job record, including its backend and
            backend-native ``external_id``.

        Raises:
            RuntimeError: If the agent reports the submission did not
                succeed (no ``"submitted"`` status or missing
                ``external_id``).

        Example:
            >>> job = runtime.submit_job(
            ...     job_id="job_1", job_type="trial", name="ar-study-trial-0001",
            ...     command=["python", "worker.py"], workdir=Path("."),
            ...     stdout=Path("out.log"), stderr=Path("err.log"),
            ...     resources=Resources(gpus=1),
            ... )
            >>> job.status
            'submitted'
        """
        job_metadata = {
            **(metadata or {}),
            "environment": env or {},
            "name": name,
        }
        context = {
            "job": {
                "job_id": job_id,
                "job_type": job_type,
                "name": name,
                "command": command,
                "workdir": str(workdir),
                "stdout": str(stdout),
                "stderr": str(stderr),
                "script_path": str(stdout.with_name("job.sh")),
                "environment": env or {},
                "resource_limits": resources.__dict__,
                "metadata": job_metadata,
            }
        }
        result = self._run_submission_action("submit", context)
        if result.get("status") != "submitted" or not result.get("external_id"):
            raise RuntimeError(
                f"Job submission failed: {result.get('message', result)}"
            )
        backend = str(result.get("backend") or "custom")
        record = JobRecord(
            job_id=job_id,
            study_id=self.study_id,
            trial_id=trial_id,
            node_run_id=node_run_id,
            backend=backend,
            external_id=str(result["external_id"]),
            status="submitted",
            command_json=json.dumps(
                {"command": command, "metadata": job_metadata},
                ensure_ascii=False,
                default=str,
            ),
            resources_json=json.dumps(resources.__dict__, ensure_ascii=False),
            workdir=str(workdir),
            stdout_path=str(stdout),
            stderr_path=str(stderr),
            job_type=job_type,
            created_at=get_utc_now(),
            updated_at=get_utc_now(),
        )
        self.state.create_job(record)
        return record

    def _reconcile_terminal_job(self, job: JobRecord, status: str) -> None:
        """Finish a job's node run once its backend job reaches a terminal status.

        Translates the backend's terminal status (``"completed"``/
        ``"timeout"``/other-as-failure) into a ``StageResult``, unless the
        trial was already marked ``"early_stopped"`` by the keypoint agent,
        in which case that takes priority. No-ops if the job has no
        associated node run, or that node run is no longer pending/running
        (already reconciled).

        Args:
            job: The job whose backend status just became terminal.
            status: The backend-reported terminal status (e.g.
                ``"completed"``, ``"timeout"``, ``"failed"``,
                ``"cancelled"``).
        """
        if not job.node_run_id:
            return
        node_run = self.state.get_node_run(job.node_run_id)
        if not node_run or node_run["status"] not in {"pending", "running"}:
            return

        if job.trial_id:
            trial = self.state.get_trial(job.trial_id)
            if trial and trial.status == "early_stopped":
                result = StageResult(
                    status="early_stopped",
                    message="Trial early stopped by keypoint agent",
                    payload={"trial_id": job.trial_id},
                )
            elif status == "completed":
                result = StageResult(status="success", message="External job completed")
            elif status == "timeout":
                trial.status = "timeout"
                self.state.update_trial(trial)
                result = StageResult(status="timeout", message="External job timed out")
            else:
                trial.status = "failed"
                self.state.update_trial(trial)
                result = StageResult(
                    status="failure", message=f"External job ended with {status}"
                )
        else:
            if status == "completed":
                result = StageResult(status="success", message="External job completed")
            elif status == "timeout":
                result = StageResult(status="timeout", message="External job timed out")
            else:
                result = StageResult(
                    status="failure", message=f"External job ended with {status}"
                )

        self.state.finish_node_run(job.node_run_id, result)
        self.state.append_event(
            self.study_id,
            "node_finished",
            {
                "run_id": job.node_run_id,
                "node_name": node_run["node_name"],
                "trial_id": node_run["trial_id"],
                "status": result.status,
                "message": result.message,
            },
        )
        self._schedule_next(
            current_node=node_run["node_name"],
            result=result,
            current_trial_id=node_run["trial_id"],
            controller_mode=True,
        )

    def _advance_local_ready_nodes(self, trial_id: str | None = None) -> None:
        """Run every pending ``execution: local`` node run matching ``trial_id``.

        Args:
            trial_id: If given, only advance that trial's local node runs;
                ``None`` only advances study-level (no-trial) node runs.
        """
        for node_run in self.state.get_pending_node_runs(self.study_id):
            if trial_id is None:
                if node_run["trial_id"] is not None:
                    continue
            elif node_run["trial_id"] != trial_id:
                continue
            node_cfg = self.graph.nodes[node_run["node_name"]]
            if node_cfg.execution != "local":
                continue
            self.run_node(
                node_name=node_run["node_name"],
                run_id=node_run["run_id"],
                trial_id=node_run["trial_id"],
                attempt_id=f"attempt_{int(node_run['attempt']):06d}",
                controller_mode=True,
            )
            self._submit_pending_trial_nodes()

    @staticmethod
    def _parse_training_progress(
        metrics_csv_path: Path, columns: tuple[str, ...]
    ) -> int | None:
        """Return the latest populated integer progress value from metrics.csv."""
        try:
            with open(
                metrics_csv_path, newline="", encoding="utf-8", errors="replace"
            ) as f:
                rows = list(csv.DictReader(f))
            # Lightning can emit rows with only train or validation fields.  Search
            # backwards rather than assuming the last row has every progress field.
            for row in reversed(rows):
                for col in columns:
                    val = row.get(col)
                    if val not in ("", None):
                        return int(float(val))
        except Exception:
            pass
        return None

    def _parse_training_epoch(self, metrics_csv_path: Path) -> int | None:
        """Return the latest epoch integer from a Lightning metrics CSV."""
        return self._parse_training_progress(metrics_csv_path, ("epoch", "Epoch"))

    def _parse_training_iteration(self, metrics_csv_path: Path) -> int | None:
        """Return the latest optimizer iteration from a Lightning metrics CSV."""
        return self._parse_training_progress(
            metrics_csv_path, ("step", "global_step", "Step", "GlobalStep")
        )

    @staticmethod
    def _compute_elapsed_trial_seconds(started_at: str) -> int | None:
        """Return elapsed seconds since the trial worker entered the running state."""
        try:
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            return max(0, int((datetime.now(timezone.utc) - started).total_seconds()))
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _load_checked_keypoints(path: Path) -> dict[str, list[int]]:
        """Load keypoint history, accepting the pre-trigger epoch-only format."""
        empty = {"epoch": [], "iteration": [], "elapsed_time": []}
        if not path.exists():
            return empty
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return empty
        if isinstance(data, list):
            # Version 1 recorded only epoch thresholds as a plain JSON array.
            empty["epoch"] = [int(value) for value in data]
            return empty
        if not isinstance(data, dict):
            return empty
        for kind in empty:
            values = data.get(kind, [])
            if isinstance(values, list):
                empty[kind] = [int(value) for value in values]
        return empty

    def _collect_keypoint_comparison(self, trial_id: str) -> str:
        """Gather CSV tails from other trials for comparison context."""
        trials_dir = self.study_dir / "trials"
        parts: list[str] = []
        for other_dir in sorted(trials_dir.iterdir()):
            if not other_dir.is_dir() or other_dir.name == trial_id:
                continue
            candidates = sorted(
                other_dir.glob("exp/csv_logs/**/metrics.csv"),
                key=lambda p: p.stat().st_mtime,
            )
            if not candidates:
                continue
            try:
                with open(
                    candidates[-1], newline="", encoding="utf-8", errors="replace"
                ) as f:
                    rows = list(csv.reader(f))
                if not rows:
                    continue
                header = ",".join(rows[0])
                tail = "\n".join(",".join(r) for r in rows[-5:])
                parts.append(f"{other_dir.name}:\n{header}\n{tail}")
            except Exception:
                continue
        return "\n\n".join(parts) if parts else "No other trial metrics available."

    def _call_keypoint_agent(
        self,
        trial,
        prompt: str,
        artifact_dir: Path,
    ) -> dict | None:
        """Request a validated stop decision through the configured agent provider."""
        request = AgentRequest(
            task="early_stop",
            objective=prompt,
            knowledge="",
            repo_context="",
            trial_history_csv="",
            latest_metrics={},
            context_files={"early_stop_instructions": prompt},
            allowed_actions=["read_only"],
            output_schema=load_schema("early_stop"),
            resume_thread_id=getattr(trial, "codex_thread_id", None),
        )
        response = self.agent.run(request, artifact_dir=artifact_dir)
        if response.status != "success":
            self.logger.warning(
                "Invalid early-stop decision for %s; leaving keypoint unchecked.",
                trial.trial_id,
            )
            return None
        result = dict(response.structured)
        result["key_metrics"] = convert_pairs_to_dict(result["key_metrics"])
        (artifact_dir / "decision.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    def _stop_trial_early(
        self,
        trial,
        trial_dir: Path,
        decision: dict,
    ) -> None:
        """Write EARLY_STOPPED.md, mark the trial, and ask its GPU job to stop."""
        reason = str(decision.get("reason", "No reason provided"))
        key_metrics = decision.get("key_metrics", {})
        marker = (
            f"# Early Stopped\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Key Metrics:**\n"
            f"```json\n{json.dumps(key_metrics, ensure_ascii=False, indent=2)}\n```\n"
        )
        (trial_dir / "EARLY_STOPPED.md").write_text(marker, encoding="utf-8")
        trial.status = "early_stopped"
        self.state.update_trial(trial)
        self.logger.info(f"Early stopping {trial.trial_id}: {reason[:120]}")

        # The GPU job's in-process monitor sees this file and stops its training
        # child without ending the job, which then carries on as the controller.
        (trial_dir / "EARLY_STOP_REQUESTED").write_text(reason, encoding="utf-8")

    def _maybe_check_keypoints(self, trial_id: str) -> None:
        """Ask the agent at configured epoch, iteration, or elapsed-time keypoints."""
        # Respect config flag
        early_stop_cfg = getattr(self.config.atlas, "early_stopping", None)
        if early_stop_cfg is not None and not bool(
            getattr(early_stop_cfg, "enabled", True)
        ):
            return

        trial = self.state.get_trial(trial_id)
        if trial.status != "running":
            return
        trial_dir = self.study_dir / "trials" / trial_id

        # Epoch keypoints preserve the original default: ten checks over a run.
        train_cfg_path = trial_dir / "resolved_training_config.yaml"
        total_epochs = 0
        if train_cfg_path.exists():
            try:
                import yaml

                with open(train_cfg_path, encoding="utf-8") as f:
                    train_cfg = yaml.safe_load(f) or {}
                total_epochs = int(
                    (train_cfg.get("trainer") or {}).get("max_epochs") or 0
                )
            except Exception:
                pass

        epoch_interval_value = getattr(early_stop_cfg, "epoch_interval", None)
        if epoch_interval_value is None:
            epoch_interval = max(1, round(total_epochs / 10)) if total_epochs > 0 else 0
        else:
            epoch_interval = max(0, int(epoch_interval_value or 0))
        iteration_interval = max(
            0, int(getattr(early_stop_cfg, "iteration_interval", 0) or 0)
        )
        elapsed_interval = max(
            0, int(getattr(early_stop_cfg, "elapsed_time_interval_sec", 0) or 0)
        )
        if not any((epoch_interval, iteration_interval, elapsed_interval)):
            return

        # Find metrics CSV
        candidates = sorted(
            trial_dir.glob("exp/csv_logs/**/metrics.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            return
        metrics_csv = candidates[-1]

        current_epoch = self._parse_training_epoch(metrics_csv)
        current_iteration = self._parse_training_iteration(metrics_csv)
        elapsed_seconds = self._compute_elapsed_trial_seconds(trial.updated_at)

        due_keypoints: list[tuple[str, int, str]] = []
        if (
            epoch_interval
            and current_epoch is not None
            and current_epoch >= epoch_interval
        ):
            target = (current_epoch // epoch_interval) * epoch_interval
            due_keypoints.append(("epoch", target, f"epoch {target}/{total_epochs}"))
        if (
            iteration_interval
            and current_iteration is not None
            and current_iteration >= iteration_interval
        ):
            target = (current_iteration // iteration_interval) * iteration_interval
            due_keypoints.append(("iteration", target, f"iteration {target}"))
        if (
            elapsed_interval
            and elapsed_seconds is not None
            and elapsed_seconds >= elapsed_interval
        ):
            target = (elapsed_seconds // elapsed_interval) * elapsed_interval
            due_keypoints.append(
                ("elapsed_time", target, f"elapsed GPU runtime {target}s")
            )
        if not due_keypoints:
            return

        kp_file = trial_dir / "keypoints_checked.json"
        with FileLock(trial_dir / ".lock_keypoints"):
            checked = self._load_checked_keypoints(kp_file)
            unchecked = [
                keypoint
                for keypoint in due_keypoints
                if keypoint[1] not in checked[keypoint[0]]
            ]
            if not unchecked:
                return

            progress = [
                (
                    f"epoch={current_epoch}/{total_epochs}"
                    if current_epoch is not None
                    else "epoch=unknown"
                ),
                (
                    f"iteration={current_iteration}"
                    if current_iteration is not None
                    else "iteration=unknown"
                ),
                (
                    f"elapsed_gpu_runtime_sec={elapsed_seconds}"
                    if elapsed_seconds is not None
                    else "elapsed_gpu_runtime_sec=unknown"
                ),
            ]
            prompt = load_prompt(
                "early_stop",
                trial_id=trial_id,
                labels=", ".join(label for _, _, label in unchecked),
                progress=", ".join(progress),
                trial_dir=trial_dir,
                metrics_csv=metrics_csv,
                study_dir=self.study_dir,
            )

            primary_kind, primary_target, _ = unchecked[0]
            kp_dir = trial_dir / "keypoints" / f"{primary_kind}_{primary_target:06d}"
            kp_dir.mkdir(parents=True, exist_ok=True)
            (kp_dir / "triggers.json").write_text(
                json.dumps(
                    {
                        "triggers": [
                            {"kind": kind, "target": target, "label": label}
                            for kind, target, label in unchecked
                        ],
                        "current_epoch": current_epoch,
                        "current_iteration": current_iteration,
                        "elapsed_gpu_runtime_sec": elapsed_seconds,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            self.logger.info(
                f"Keypoint check for {trial_id}: "
                f"{', '.join(label for _, _, label in unchecked)}"
            )
            result = self._call_keypoint_agent(trial, prompt, kp_dir)
            if result is None:
                return

            for kind, target, _ in unchecked:
                checked[kind].append(target)
            kp_file.write_text(
                json.dumps(checked, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        if result and result.get("should_stop"):
            self._stop_trial_early(trial, trial_dir, result)

    def _maybe_schedule_parallel_proposals(self) -> None:
        """Create enough new ``propose_trial`` node runs to fill open parallel slots.

        Only applies to graphs that start at ``propose_trial`` (i.e.
        continuous proposal/run applications, not a from-scratch init). Under
        a study-wide lock, computes how many trial "slots" are already
        occupied by active trial jobs, queued ``run_trial`` nodes, and
        in-flight ``propose_trial``/``debug_trial`` node runs, then creates
        new ``propose_trial`` node runs for however many slots remain open,
        capped by ``budget.max_trials`` if set.
        """
        if "propose_trial" not in self.graph.nodes:
            return
        if self.get_bootstrap_node_name() != "propose_trial":
            return
        if self.should_stop():
            return
        budget = self.config.atlas.budget
        max_parallel = int(getattr(budget, "max_parallel_trials", 1) or 1)
        max_trials = int(getattr(budget, "max_trials", 0) or 0)
        with FileLock(self.study_dir / ".lock_proposal_scheduler"):
            trials = self.state.list_trials(self.study_id)
            node_runs = self.state.list_node_runs(self.study_id)
            active_trial_jobs = sum(
                1
                for job in self.state.list_active_jobs(self.study_id)
                if job.job_type == "trial"
            )
            queued_run_trials = sum(
                1
                for node_run in node_runs
                if node_run["node_name"] == "run_trial"
                and node_run["status"] == "pending"
                and self.state.find_job_by_node_run(node_run["run_id"]) is None
            )
            inflight_proposals = sum(
                1
                for node_run in node_runs
                if node_run["node_name"] == "propose_trial"
                and node_run["status"] in {"pending", "running"}
            )
            inflight_debug_trials = sum(
                1
                for node_run in node_runs
                if node_run["node_name"] == "debug_trial"
                and node_run["status"] in {"pending", "running"}
            )
            occupied_slots = (
                active_trial_jobs
                + queued_run_trials
                + inflight_proposals
                + inflight_debug_trials
            )
            open_slots = max_parallel - occupied_slots
            if open_slots <= 0:
                return
            remaining_budget = open_slots
            if max_trials > 0:
                reserved_trials = len(trials) + inflight_proposals
                remaining_budget = min(remaining_budget, max_trials - reserved_trials)
            for _ in range(max(0, remaining_budget)):
                run_id = f"run_{uuid.uuid4().hex}"
                self.state.create_node_run(
                    run_id=run_id,
                    study_id=self.study_id,
                    node_name="propose_trial",
                )
                self.state.append_event(
                    self.study_id,
                    "node_scheduled",
                    {"run_id": run_id, "node_name": "propose_trial", "trial_id": None},
                )

    def _has_completed_training(self, trial_id: str) -> bool:
        """Allow evaluation-only retries to share capacity with new training trials."""
        path = self.study_dir / "trials" / trial_id / "stage_status.yaml"
        if not path.is_file():
            return False
        try:
            status = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
            return status["stages"]["train"]["status"] == "completed"
        except (KeyError, TypeError, ValueError):
            return False

    def _find_inflight_retry_trial_ids(
        self, node_runs: list[dict] | None = None
    ) -> set[str]:
        """Return trials currently in debug, repair, or retry execution."""
        if node_runs is None:
            node_runs = self.state.list_node_runs(self.study_id)
        retry_trial_ids = set()
        active_statuses = {"pending", "running"}
        for node_run in node_runs:
            trial_id = node_run.get("trial_id")
            if not trial_id or node_run.get("status") not in active_statuses:
                continue
            node_name = node_run.get("node_name")
            if node_name == "debug_trial":
                retry_trial_ids.add(str(trial_id))
            elif node_name in {"implement_code", "run_tests"}:
                trial = self.state.get_trial(str(trial_id))
                if trial.attempt_count > 0:
                    retry_trial_ids.add(str(trial_id))
        # A resumed user-paused run may have attempt > 1 without being failure
        # recovery. Actual failures set recovery_active in _schedule_next;
        # keep those blocked until recovery completes, not every resubmission.
        retry_trial_ids.update(
            trial.trial_id
            for trial in self.state.list_trials(self.study_id)
            if trial.recovery_active
            and not self._has_completed_training(trial.trial_id)
        )
        return retry_trial_ids

    def _submit_pending_trial_nodes(self) -> None:
        """Submit pending trial nodes under one study-wide scheduler lock."""
        if (self.study_dir / "STOP").exists():
            return
        with FileLock(self.study_dir / ".lock_trial_submit"):
            self._submit_pending_trial_nodes_locked()

    def _submit_pending_trial_nodes_locked(self) -> None:
        """Submit pending ``execution: submission`` node runs, up to the free slots.

        Must be called while holding ``.lock_trial_submit`` (see
        :meth:`_submit_pending_trial_nodes`). Iterates pending node runs in
        order, skipping ``execution: local`` nodes (handled elsewhere) and
        nodes that already have a submitted job, and stops once active trial
        jobs reach ``budget.max_parallel_trials``.
        """
        budget = self.config.atlas.budget
        max_parallel = int(getattr(budget, "max_parallel_trials", 1) or 1)
        active_trial_jobs = sum(
            1
            for job in self.state.list_active_jobs(self.study_id)
            if job.node_run_id
            and self.state.get_node_run(job.node_run_id)["node_name"] == "run_trial"
        )
        for node_run in self.state.get_pending_node_runs(self.study_id):
            node_cfg = self.graph.nodes[node_run["node_name"]]
            if node_cfg.execution == "local":
                continue
            if active_trial_jobs >= max_parallel:
                return
            if self.state.find_job_by_node_run(node_run["run_id"]) is not None:
                continue
            trial_id = node_run["trial_id"]
            attempt = int(node_run["attempt"] or 0)
            attempt_id = f"attempt_{attempt:06d}"
            workdir = self.recipe_dir
            marker_dir = self.study_dir / "node_runs" / node_run["run_id"]
            if trial_id:
                attempt_dir = (
                    self.study_dir / "trials" / trial_id / "attempts" / attempt_id
                )
            else:
                attempt_dir = marker_dir
            stdout = attempt_dir / "stdout.log"
            stderr = attempt_dir / "stderr.log"
            command = self._build_node_worker_command(
                node_name=node_run["node_name"],
                run_id=node_run["run_id"],
                trial_id=trial_id,
                attempt_id=attempt_id,
            )
            metadata = {"marker_dir": str(marker_dir)}
            resources = self.trial_resources_default.merge_with(node_cfg.resources)
            env = {
                "PYTHONPATH": os.pathsep.join(
                    path
                    for path in (
                        str(Path(__file__).resolve().parents[4]),
                        os.environ.get("PYTHONPATH", ""),
                    )
                    if path
                ),
            }
            job_id = f"job_{uuid.uuid4().hex}"
            job = self.submit_job(
                job_id=job_id,
                job_type="trial",
                name=self._build_job_name(
                    node_name=node_run["node_name"],
                    trial_id=trial_id,
                    attempt_id=attempt_id,
                ),
                command=command,
                workdir=workdir,
                stdout=stdout,
                stderr=stderr,
                resources=resources,
                env=env,
                metadata=metadata,
                trial_id=trial_id,
                node_run_id=node_run["run_id"],
            )
            active_trial_jobs += 1
            self.state.append_event(
                self.study_id,
                "job_submitted",
                {
                    "run_id": node_run["run_id"],
                    "node_name": node_run["node_name"],
                    "trial_id": trial_id,
                    "backend": job.backend,
                    "external_id": job.external_id,
                    "command": " ".join(shlex.quote(part) for part in command),
                },
            )

    def tick(self, trial_id: str | None = None) -> None:
        """Run one scheduler cycle: bootstrap, reconcile jobs, advance.

        Bootstraps the study's first node run if none exist yet, refreshes
        active job statuses (reconciling any that finished), schedules new parallel
        proposals and submits pending submission-execution nodes (only for a
        whole-study tick, i.e. ``trial_id is None``), and advances any
        pending local-execution nodes. No-ops entirely if the study's
        ``STOP`` file exists.

        Args:
            trial_id: If given, scope the refresh/local-node work to this
                trial only (used by a GPU self-controller acting for just its
                own trial); ``None`` runs the full study-wide tick.

        Returns:
            None.

        Example:
            >>> runtime.tick()
        """
        if (self.study_dir / "STOP").exists():
            return
        if trial_id is None and not self.state.list_node_runs(self.study_id):
            self.state.create_node_run(
                run_id=f"run_{uuid.uuid4().hex}",
                study_id=self.study_id,
                node_name=self.get_bootstrap_node_name(),
            )
        self._refresh_active_jobs(
            trial_id=trial_id,
            # A controller tick must reconcile trial jobs as well.  The old
            # condition left completed Slurm jobs marked active unless their
            # per-trial monitor ran, which is unavailable in self-controller
            # mode.
            include_trial_jobs=True,
        )
        if trial_id is None:
            self._maybe_schedule_parallel_proposals()
        self._submit_pending_trial_nodes()
        self._advance_local_ready_nodes(trial_id=trial_id)
        self._submit_pending_trial_nodes()
