"""Shared Atlas data models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from omegaconf import DictConfig

from espnet3.bin.atlas.core.errors import GraphRoutingError
from espnet3.bin.atlas.core.serialization import dumps_json, loads_json

if TYPE_CHECKING:
    from espnet3.bin.atlas.core.graph import ResearchGraph
    from espnet3.bin.atlas.core.scheduler import AtlasGraphRuntime
    from espnet3.bin.atlas.core.state import StateStore


@dataclass
class Study:
    """Study record."""

    study_id: str
    study_dir: str
    status: str
    created_at: str
    updated_at: str
    current_best_trial_id: str | None = None
    agent_session_id: str | None = None
    agent_context_initialized: bool = False


@dataclass
class JobRecord:
    """Persistent job record."""

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
    """Resource limits passed to a submission agent."""

    gpus: int = 0
    cpus: int = 1
    nodes: int = 1
    mem: str | None = None
    time: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config) -> "Resources":
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

    def merged_with(self, override: "Resources | None") -> "Resources":
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
    """Per-node graph config."""

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
    fallback_output: str | None = None
    on: dict[str, str] = field(default_factory=dict)
    terminal: bool = False


@dataclass
class ResearchGraph:
    """Static stage graph."""

    name: str
    version: int
    start: str
    nodes: dict[str, NodeConfig]

    def route(self, node_name: str, status: str) -> str | None:
        node = self.nodes[node_name]
        if node.terminal:
            return None
        next_node = node.on.get(status)
        if next_node is None:
            raise GraphRoutingError(
                f"Node '{node_name}' has no routing for status '{status}'. "
                f"Defined: {sorted(node.on.keys())}"
            )
        return next_node


@dataclass
class AgentRequest:
    """Request sent to a coding agent."""

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
    """Structured response from a coding agent."""

    status: Literal["success", "failure"]
    message: str
    content: str
    structured: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    thread_id: str | None = None


@dataclass
class CodexConfig:
    """Options for ``codex exec``."""

    executable: str = "codex"
    model: str | None = None
    sandbox: str | None = "workspace-write"
    skip_git_repo_check: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)


@dataclass
class ClaudeConfig:
    """Options for Claude Code's print mode."""

    executable: str = "claude"
    model: str | None = None
    permission_mode: str = "bypassPermissions"
    effort: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    max_turns: int | None = None
    args: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Configuration for the unified CLI agent."""

    provider: Literal["codex", "claude"]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_sec: float | None = None
    session_id: str | None = None
    structured_output_retries: int = 2
    codex: CodexConfig = field(default_factory=CodexConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)

    @classmethod
    def from_config(cls, config) -> "AgentConfig":
        """Build a validated agent config from a mapping or OmegaConf node.

        Args:
            config: Mapping or OmegaConf node with provider and CLI options.

        Returns:
            AgentConfig: Parsed configuration for Codex or Claude.

        Example:
            >>> AgentConfig.from_config({"provider": "codex"}).provider
            'codex'
        """
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(config):
                config = OmegaConf.to_container(config, resolve=True)
        except ImportError:
            pass
        values = dict(config or {})
        provider = str(values.pop("provider", "")).lower()
        if provider not in {"codex", "claude"}:
            raise ValueError("atlas.agent.provider must be 'codex' or 'claude'")
        values["provider"] = provider
        if "codex" in values:
            codex_values = dict(values["codex"] or {})
            codex_values["args"] = [str(arg) for arg in codex_values.get("args", [])]
            codex_values["config"] = dict(codex_values.get("config", {}) or {})
            values["codex"] = CodexConfig(**codex_values)
        if "claude" in values:
            claude_values = dict(values["claude"] or {})
            for name in ("args", "allowed_tools", "disallowed_tools"):
                claude_values[name] = [str(arg) for arg in claude_values.get(name, [])]
            values["claude"] = ClaudeConfig(**claude_values)
        values["env"] = {
            str(k): str(v) for k, v in dict(values.get("env", {}) or {}).items()
        }
        return cls(**values)


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
    """Represents one experiment proposal."""

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
    """Returned by every Atlas stage."""

    status: StageStatus
    message: str = ""
    next_node: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "next_node": self.next_node,
            "payload": self.payload,
            "artifacts": self.artifacts,
        }

    def to_json(self) -> str:
        return dumps_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "StageResult":
        return cls(**loads_json(text))


@dataclass
class StageContext:
    """Runtime context shared with a stage."""

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

    def trial_dir(self) -> Path:
        assert self.current_trial is not None
        return self.study_dir / "trials" / self.current_trial.trial_id

    def artifact_dir(self, *parts: str) -> Path:
        return self.study_dir / "node_runs" / self.run_id / Path(*parts)
