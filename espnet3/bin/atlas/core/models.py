"""Shared Atlas data models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from omegaconf import DictConfig

from espnet3.bin.atlas.core import AtlasError

if TYPE_CHECKING:
    from espnet3.bin.atlas.core.scheduler import AtlasGraphRuntime
    from espnet3.bin.atlas.core.state import StateStore


@dataclass
class Study:
    """One row of a study's persistent record, as read from/written to state.

    Args:
        study_id: The study's name (``atlas.study_name``), used as its key.
        study_dir: Absolute path to the study directory.
        status: Study lifecycle status, e.g. ``"initialized"``, ``"running"``,
            ``"stopped"``, ``"completed"``.
        created_at: ISO-8601 UTC timestamp the study was first created.
        updated_at: ISO-8601 UTC timestamp of the last state change.
        current_best_trial_id: Trial id of the current best trial, or
            ``None`` if none has been accepted yet.
        agent_session_id: The coding agent's resumable session/thread id for
            this study, or ``None`` before the first agent call.
        is_agent_context_initialized: Whether the agent has already been given
            the study's full context (objective, knowledge, repo context) in
            this session.
    """

    study_id: str
    study_dir: str
    status: str
    created_at: str
    updated_at: str
    current_best_trial_id: str | None = None
    agent_session_id: str | None = None
    is_agent_context_initialized: bool = False


@dataclass
class JobRecord:
    """One row of a submitted job's persistent record (Slurm or local process).

    Args:
        job_id: Atlas-assigned id for this job.
        study_id: Study the job belongs to.
        trial_id: Trial the job is running, or ``None`` for a study-level
            job (such as a controller tick).
        node_run_id: Graph node run the job executes, or ``None``.
        backend: Submission backend, e.g. ``"slurm"`` or ``"local"``.
        external_id: Backend-native job id (e.g. the Slurm job id), or
            ``None`` before submission completes.
        status: Job status such as ``"pending"``, ``"running"``,
            ``"completed"``, ``"failed"``.
        command_json: JSON-encoded command argv that was submitted.
        resources_json: JSON-encoded ``Resources`` requested, or ``None``.
        workdir: Working directory the job runs in, or ``None``.
        stdout_path: Path the job's stdout is redirected to, or ``None``.
        stderr_path: Path the job's stderr is redirected to, or ``None``.
        job_type: Kind of job, e.g. ``"node"`` or ``"tick"`` (controller).
        created_at: ISO-8601 UTC timestamp the job record was created.
        updated_at: ISO-8601 UTC timestamp of the last status change.
    """

    job_id: str
    study_id: str
    trial_id: str | None
    node_run_id: str | None
    backend: str
    external_id: str | None
    status: str
    command_json: str
    resources_json: str | None
    workdir: str | None
    stdout_path: str | None
    stderr_path: str | None
    job_type: str
    created_at: str
    updated_at: str


@dataclass
class Resources:
    """Compute resource limits requested for a submitted job.

    Args:
        gpus: Number of GPUs to request.
        cpus: Number of CPU cores to request.
        nodes: Number of nodes to request.
        mem: Memory limit string (e.g. ``"32G"``), or ``None`` for the
            backend's default.
        time: Wall-clock time limit string (e.g. ``"24:00:00"``), or
            ``None`` for the backend's default.
        extra: Backend-specific extra fields (e.g. Slurm ``partition``,
            ``account``, ``reservation``) passed through as-is.
    """

    gpus: int = 0
    cpus: int = 1
    nodes: int = 1
    mem: str | None = None
    time: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config) -> "Resources":
        """Build ``Resources`` from an OmegaConf node or mapping.

        Known fields (``gpus``, ``cpus``, ``nodes``, ``mem``, ``time``) are
        read with their dataclass defaults as fallback; every other key
        (and any pre-existing ``extra`` mapping) is collected into ``extra``.

        Args:
            config: An OmegaConf node, object with attributes, or mapping of
                resource settings; ``None`` yields the all-defaults instance.

        Returns:
            Resources: The parsed resource limits.

        Example:
            >>> Resources.from_config({"gpus": 2, "partition": "gpuA100x4"})
            Resources(gpus=2, cpus=1, nodes=1, mem=None, time=None,
                      extra={'partition': 'gpuA100x4'})
        """
        if config is None:
            return cls()
        values = dict(config) if hasattr(config, "items") else vars(config)
        fields = ("gpus", "cpus", "nodes", "mem", "time")
        known = {key: values.get(key) for key in fields}
        extra = dict(values.get("extra", {}) or {})
        extra.update(
            {
                key: value
                for key, value in values.items()
                if key not in (*fields, "extra")
            }
        )
        return cls(
            gpus=int(known["gpus"] or 0),
            cpus=int(known["cpus"] or 1),
            nodes=int(known["nodes"] or 1),
            mem=known["mem"],
            time=known["time"],
            extra=extra,
        )

    def merge_with(self, override: "Resources | None") -> "Resources":
        """Layer a node-level resource override onto this (study-level) default.

        ``cpus``/``nodes`` are only taken from ``override`` when they differ
        from the dataclass default (``1``), since a node that does not set
        them should inherit the default rather than silently overriding to
        ``1``. ``gpus``/``mem``/``time`` are taken from ``override`` whenever
        truthy. ``extra`` is merged, with ``override`` taking priority on key
        collisions.

        Args:
            override: Node-specific resources to layer on top, or ``None``
                to keep this instance's values unchanged (returns a copy).

        Returns:
            Resources: The merged resource limits.

        Example:
            >>> Resources(gpus=1).merge_with(Resources(gpus=2))
            Resources(gpus=2, cpus=1, nodes=1, mem=None, time=None, extra={})
        """
        if override is None:
            return Resources(
                self.gpus, self.cpus, self.nodes, self.mem, self.time, dict(self.extra)
            )
        return Resources(
            gpus=override.gpus or self.gpus,
            cpus=override.cpus if override.cpus != 1 else self.cpus,
            nodes=override.nodes if override.nodes != 1 else self.nodes,
            mem=override.mem or self.mem,
            time=override.time or self.time,
            extra={**self.extra, **override.extra},
        )


@dataclass
class NodeConfig:
    """One node of a loaded graph YAML, as parsed by ``load_graph``.

    Args:
        name: Node's key in the graph's ``nodes`` mapping.
        target: Dotted ``module.Class`` path of the ``BaseStage``
            subclass to run. Required; a skill node names ``SkillStage``.
        execution: ``"local"`` to run in the current process, or
            ``"submission"`` to be dispatched as a separate submitted job.
        resources: Resource limits to request when ``execution ==
            "submission"``, or ``None`` to use the study's default.
        skill: Name of the skill prompt/markdown to run via ``SkillStage``,
            or ``None`` for a non-skill stage target.
        skill_context: Extra key/value context merged into the skill's
            ``AgentRequest.context_files``.
        output_schema: JSON Schema (with a ``type``) of each extra reply field
            of a skill node, by field name. ``status`` and ``message`` are
            added for every skill node.
        allowed_actions: Agent action set allowed for this node (e.g.
            ``["read_only"]`` or including ``"terminal"``).
        trial_update_keys: Agent response keys that are copied onto the
            current trial's matching attributes.
        output_file: Relative path (under ``study_dir``) the skill's
            ``output_key`` result is written to, or ``None``.
        output_key: Key read from the skill's structured response and
            written to ``output_file``, or ``None``.
        output_append: Whether ``output_key``'s text is appended to
            ``output_file`` instead of replacing it.
        fallback_output: Text written to ``output_file`` if the skill's
            agent call fails, or ``None`` to fail instead.
        on: Mapping from a stage result status to the next node's name.
        terminal: Whether this node ends the graph (no further routing).
    """

    name: str
    target: str
    execution: str
    resources: Resources | None
    skill: str | None = None
    skill_context: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    allowed_actions: list[str] = field(default_factory=list)
    trial_update_keys: list[str] = field(default_factory=list)
    output_file: str | None = None
    output_key: str | None = None
    output_append: bool = False
    fallback_output: str | None = None
    on: dict[str, str] = field(default_factory=dict)
    terminal: bool = False


@dataclass
class ResearchGraph:
    """A study's static stage graph, as loaded from ``graph.yaml``.

    Args:
        name: Graph's declared name.
        version: Graph schema version.
        start: Name of the first node to run for a new study.
        nodes: Mapping of node name to its ``NodeConfig``.
    """

    name: str
    version: int
    start: str
    nodes: dict[str, NodeConfig]

    def route(self, node_name: str, status: str) -> str | None:
        """Return the next node's name for a stage result, or ``None`` if terminal.

        Args:
            node_name: Node that just ran.
            status: The stage result's ``status`` value (e.g. ``"success"``).

        Returns:
            str | None: The next node's name from ``node_name``'s ``on``
            mapping, or ``None`` if the node is ``terminal``.

        Raises:
            AtlasError: If the node is not terminal and ``status``
                has no entry in its ``on`` mapping.

        Example:
            >>> graph.route("collect_result", "success")
            'accept_or_reject'
        """
        node = self.nodes[node_name]
        if node.terminal:
            return None
        next_node = node.on.get(status)
        if next_node is None:
            raise AtlasError(
                f"Node '{node_name}' has no routing for status '{status}'. "
                f"Defined: {sorted(node.on.keys())}"
            )
        return next_node


@dataclass
class AgentRequest:
    """One task sent to the configured coding agent (Codex or Claude Code).

    Built by ``SkillStage.run`` from the study's objective, knowledge pack,
    trial history, and the node's skill context.

    Args:
        task: Skill/task name the agent should perform.
        objective: Study objective text (usually ``program.md``).
        knowledge: Accumulated knowledge-pack text, or "" if none yet.
        repo_context: Cached repository-context text, or "" if none yet.
        trial_history_csv: Current ``trials.csv`` content, or "".
        latest_metrics: The current trial's metrics dict, or ``{}``.
        context_files: Extra named text/context passed to the agent (study
            dir, trial dir, serialized current trial, etc.).
        allowed_actions: Action set the agent may use (e.g. whether it may
            edit files or only read).
        output_schema: JSON Schema of the structured response the agent
            must return; empty for a free-form reply.
        inflight_patches: Config patches from trials still running, so the
            agent avoids proposing the same change twice.
        resume_thread_id: Prior Codex/Claude thread id to resume, or
            ``None`` to start a new conversation.
    """

    task: str
    objective: str
    knowledge: str
    repo_context: str
    trial_history_csv: str
    latest_metrics: dict[str, Any]
    context_files: dict[str, str]
    allowed_actions: list[str]
    output_schema: dict[str, Any]
    inflight_patches: list[dict] = field(default_factory=list)
    resume_thread_id: str | None = None


@dataclass
class AgentResponse:
    """A coding agent's parsed response to one ``AgentRequest``.

    Args:
        status: ``"success"`` if the agent completed and returned a parsable
            structured response, ``"failure"`` otherwise.
        message: Short human-readable summary of what happened.
        content: The agent's raw text output (pre-structured-output parsing).
        structured: The parsed JSON object the agent returned, matching the
            request's ``output_schema`` (e.g. ``status``/``payload``/
            ``trial_updates``), or ``{}`` if parsing failed.
        artifacts: Named output file paths the agent reported writing.
        thread_id: The agent's session/thread id, to resume in a later
            request, or ``None``.
    """

    status: Literal["success", "failure"]
    message: str
    content: str
    structured: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    thread_id: str | None = None


TrialStatus = Literal[
    "proposing",
    "proposed",
    "running",
    "completed",
    "failed",
    "timeout",
    "needs_resume",
    "accepted",
    "rejected",
    "cancelled",
    "early_stopped",
]


@dataclass
class Trial:
    """One proposed/run hyperparameter or architecture experiment.

    Args:
        trial_id: Unique id for this trial within its study.
        study_id: Study this trial belongs to.
        status: Lifecycle status, e.g. ``"proposing"``, ``"running"``,
            ``"accepted"``, ``"rejected"``, ``"failed"``.
        config_patch: Dotted-key overrides applied on top of the base
            training config to produce this trial's config.
        resolved_config_path: Path to the fully resolved config written for
            this trial, or ``None`` before it is generated.
        rationale: Agent-provided reasoning for proposing this change.
        expected_effect: Agent-provided expectation of the change's effect.
        risk: Agent-provided assessment of the change's risk.
        score: The trial's extracted metric value, or ``None`` before
            ``collect_result`` runs.
        score_name: Name of the metric ``score`` is, or ``None``.
        decision: ``"accept"``/``"reject"`` decision from
            ``accept_or_reject``, or ``None`` before it runs.
        parent_trial_id: Trial this one was proposed from, or ``None`` for
            the first trial.
        attempt_count: Number of run attempts made for this trial so far.
        created_at: ISO-8601 UTC timestamp the trial was proposed.
        updated_at: ISO-8601 UTC timestamp of the last field change.
        recovery_active: Whether a `debug_trial` recovery attempt is in
            progress for this trial.
        metrics: Full extracted metrics dict (``score`` is one entry of it).
        reflection: Agent-written reflection on this trial's outcome.
        next_suggestion: Agent-written suggestion for the next trial.
        codex_thread_id: Resumable agent thread id tied to this trial, or
            ``None``.
        extra_stages: Names of extra stages this trial still needs to run
            (for applications with per-trial stage variation).
    """

    trial_id: str
    study_id: str
    status: TrialStatus
    config_patch: dict[str, Any]
    resolved_config_path: Path | None
    rationale: str
    expected_effect: str
    risk: str
    score: float | None
    score_name: str | None
    decision: str | None
    parent_trial_id: str | None
    attempt_count: int
    created_at: str
    updated_at: str
    recovery_active: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    reflection: str = ""
    next_suggestion: str = ""
    codex_thread_id: str | None = None
    extra_stages: list[str] = field(default_factory=list)


StageStatus = Literal[
    "success",
    "failure",
    "timeout",
    "retry",
    "duplicate",
    "skip",
    "accepted",
    "rejected",
    "continue",
    "done",
    "no_budget",
    "waiting_agent",
    "waiting_job",
    "terminal",
]


@dataclass
class StageResult:
    """The outcome of one ``BaseStage.run`` call, used to route the graph.

    Persisted by ``BaseStage.execute`` as the node run's success/failure
    marker, and read back on a re-dispatched run via :meth:`from_json`.

    Args:
        status: Outcome used by ``ResearchGraph.route`` to pick the next
            node (e.g. ``"success"``, ``"failure"``, ``"timeout"``,
            ``"terminal"``).
        message: Short human-readable description of the outcome.
        next_node: Explicit next-node override (bypassing the graph's ``on``
            routing), or ``None`` to route normally.
        payload: Stage-specific result data.
        artifacts: Named output file paths the stage wrote.
    """

    status: StageStatus
    message: str = ""
    next_node: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a plain dict with all five fields.

        Args:
            None.

        Returns:
            dict[str, Any]: ``{"status", "message", "next_node", "payload",
            "artifacts"}``.

        Example:
            >>> StageResult(status="success").to_dict()["status"]
            'success'
        """
        return {
            "status": self.status,
            "message": self.message,
            "next_node": self.next_node,
            "payload": self.payload,
            "artifacts": self.artifacts,
        }

    def to_json(self) -> str:
        """Serialize this result to JSON, for the stage marker file.

        Args:
            None.

        Returns:
            str: JSON text, as produced by :func:`json.dumps` on
            :meth:`to_dict`.

        Example:
            >>> StageResult(status="success").to_json()
            '{\\n  "status": "success",\\n  ...\\n}'
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)

    @classmethod
    def from_json(cls, text: str) -> "StageResult":
        """Reconstruct a result from the JSON written by :meth:`to_json`.

        Args:
            text: JSON text previously produced by :meth:`to_json`.

        Returns:
            StageResult: The reconstructed result.

        Example:
            >>> StageResult.from_json('{"status": "success"}').status
            'success'
        """
        return cls(**json.loads(text))


@dataclass
class StageContext:
    """Everything one node run of ``BaseStage.run`` needs to do its work.

    Built fresh for each node run by the scheduler and passed to
    ``stage.execute(context)``.

    Args:
        study_dir: Study's root directory.
        recipe_dir: Recipe root directory.
        node_name: Graph node being run.
        run_id: Unique id of this node run.
        attempt_id: Attempt id within this node run (for retries).
        graph: The study's loaded ``ResearchGraph``.
        config: The study's resolved ``atlas.yaml`` config.
        state: The study's ``StateStore``.
        agent: The configured coding-agent client (e.g. ``CliAgent``).
        logger: Logger scoped to this run.
        scheduler: The owning ``AtlasGraphRuntime``.
        current_trial: The trial this node run operates on, or ``None`` for
            study-level nodes (e.g. ``research_context``).
        gpu_self_controller: Whether this node run is executing inside a GPU
            job acting as its own controller (affects some stages'
            behavior, e.g. whether to submit a follow-up job).
    """

    study_dir: Path
    recipe_dir: Path
    node_name: str
    run_id: str
    attempt_id: str
    graph: "ResearchGraph"
    config: DictConfig
    state: "StateStore"
    agent: Any
    logger: logging.Logger
    scheduler: "AtlasGraphRuntime"
    current_trial: Trial | None = None
    gpu_self_controller: bool = False

    def get_trial_dir(self) -> Path:
        """Return this run's trial directory: ``study_dir/trials/<trial_id>``.

        Args:
            None.

        Returns:
            Path: The current trial's directory.

        Raises:
            AssertionError: If called when ``current_trial`` is ``None``.

        Example:
            >>> context.get_trial_dir()
            PosixPath('exp/atlas/my_study/trials/trial_001')
        """
        assert self.current_trial is not None
        return self.study_dir / "trials" / self.current_trial.trial_id

    def get_artifact_dir(self, *parts: str) -> Path:
        """Return this node run's artifact directory, optionally with subparts.

        Args:
            *parts: Optional path components appended under the run's
                artifact directory.

        Returns:
            Path: ``study_dir/node_runs/<run_id>[/parts...]``.

        Example:
            >>> context.get_artifact_dir("skill_name")
            PosixPath('exp/atlas/my_study/node_runs/run_abc123/skill_name')
        """
        return self.study_dir / "node_runs" / self.run_id / Path(*parts)
