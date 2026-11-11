"""Atlas stage base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from pathlib import Path

from espnet3.bin.atlas.agents.interface import AgentRequest
from espnet3.bin.atlas.core.models import StageResult, Trial
from espnet3.bin.atlas.core.serialization import (
    atomic_write_text,
    dumps_json,
    file_lock,
    to_plain_data,
    utc_now,
)


class AtlasStage(ABC):
    """Base class implementing idempotent stage execution."""

    study_lock_name: str | None = None

    def execute(self, context) -> StageResult:
        marker_dir = context.artifact_dir()
        marker_dir.mkdir(parents=True, exist_ok=True)
        success_path = marker_dir / ".stage_success.json"
        failure_path = marker_dir / ".stage_failure.json"
        started_path = marker_dir / ".stage_started.json"
        if success_path.exists() and not bool(context.config.get("force", False)):
            return StageResult.from_json(success_path.read_text(encoding="utf-8"))
        atomic_write_text(started_path, f'{{"started_at": "{utc_now()}"}}')
        if self.study_lock_name is None:
            result = self.run(context)
        else:
            with file_lock(context.study_dir / f".lock_{self.study_lock_name}"):
                result = self.run(context)
        target = success_path
        if result.status in {"failure", "timeout", "retry"}:
            target = failure_path
        atomic_write_text(target, result.to_json())
        return result

    @abstractmethod
    def run(self, context) -> StageResult:
        """Run the stage."""


class SkillStage(AtlasStage):
    """Run a graph-selected skill through the configured coding agent."""

    study_lock_name = "study_controller"

    def run(self, context) -> StageResult:
        """Run the selected skill and apply its safe outputs to study state.

        Args:
            context: Runtime context containing graph, study, agent, and trial.

        Returns:
            StageResult: Skill status, payload, and any written artifacts.

        Example:
            >>> result = SkillStage().run(stage_context)
            >>> result.status
            'success'
        """
        node = context.graph.nodes[context.node_name]
        if not node.skill:
            return StageResult(
                status="failure", message="SkillStage requires a graph skill"
            )

        def read_optional(path: Path) -> str:
            return path.read_text(encoding="utf-8") if path.is_file() else ""

        trial = context.current_trial
        trial_dir = context.trial_dir() if trial else context.artifact_dir()
        allowed_actions = list(node.allowed_actions or ["read_only"])
        code_edit_config = getattr(context.config.atlas, "code_edit", None)
        if node.skill == "hyperparameter_search_implement_code" and not bool(
            getattr(code_edit_config, "allow_pixi", False)
        ):
            allowed_actions = [
                action for action in allowed_actions if action != "terminal"
            ]
        trials_csv = context.study_dir / "trials.csv"
        objective = read_optional(context.study_dir / "program.md")
        request = AgentRequest(
            task=node.skill,
            objective=objective,
            knowledge=read_optional(
                context.study_dir / "knowledge" / "knowledge_pack.md"
            ),
            repo_context=read_optional(
                context.study_dir / "knowledge" / "repo_context.md"
            ),
            trial_history_csv=read_optional(trials_csv),
            latest_metrics=trial.metrics if trial else {},
            context_files={
                "study_dir": str(context.study_dir),
                "recipe_dir": str(context.recipe_dir),
                "trial_dir": str(trial_dir),
                "current_trial": dumps_json(to_plain_data(trial)) if trial else "",
                "skill_context": dumps_json(to_plain_data(node.skill_context)),
                "code_edit_config": dumps_json(
                    to_plain_data(getattr(context.config.atlas, "code_edit", {}))
                ),
                "stop_after_trial": str(context.scheduler.should_stop()).lower(),
            },
            allowed_actions=allowed_actions,
            output_schema={
                "status": "graph result status",
                "message": "short result description",
                "payload": {},
                "trial_updates": {},
                **node.output_schema,
            },
            resume_thread_id=trial.codex_thread_id if trial else None,
        )
        artifact_dir = trial_dir / node.skill if trial else trial_dir
        output_path = None
        if node.output_file:
            relative = Path(
                node.output_file.format(trial_id=trial.trial_id if trial else "")
            )
            if relative.is_absolute() or ".." in relative.parts:
                return StageResult(
                    status="failure", message="Skill output_file must stay in study_dir"
                )
            output_path = context.study_dir / relative
        response = context.agent.run(request, artifact_dir=artifact_dir)
        if response.status != "success":
            if output_path and node.fallback_output:
                atomic_write_text(output_path, node.fallback_output)
                return StageResult(
                    status="terminal",
                    message="Wrote fallback output after skill agent failure",
                    artifacts={"output": str(output_path)},
                )
            return StageResult(
                status="failure", message=response.message or "Skill agent failed"
            )

        result = dict(response.structured or {})
        status = str(result.pop("status", "success"))
        message = str(
            result.pop("message", result.get("summary", response.content or node.skill))
        )
        payload = dict(result.pop("payload", {}) or {})
        trial_updates = dict(result.pop("trial_updates", {}) or {})
        trial_updates.update(
            {key: result[key] for key in node.trial_update_keys if key in result}
        )
        if trial is not None:
            if response.thread_id:
                trial_updates["codex_thread_id"] = response.thread_id
            allowed_trial_fields = {field.name for field in fields(Trial)} - {
                "trial_id",
                "study_id",
                "created_at",
                "updated_at",
                "config_patch",
                "resolved_config_path",
            }
            for key, value in trial_updates.items():
                if key in allowed_trial_fields:
                    setattr(trial, key, value)
            if trial_updates:
                context.state.update_trial(trial)

        artifacts: dict[str, str] = {}
        output_key = node.output_key
        if node.output_file and output_key and result.get(output_key) is None:
            if node.fallback_output and output_path:
                atomic_write_text(output_path, node.fallback_output)
                return StageResult(
                    status="terminal",
                    message=f"Wrote fallback because skill omitted '{output_key}'",
                    artifacts={"output": str(output_path)},
                )
            return StageResult(
                status="failure",
                message=(
                    f"Skill '{node.skill}' did not return required field "
                    f"'{output_key}'"
                ),
            )
        if node.output_file and output_key and result.get(output_key) is not None:
            assert output_path is not None
            atomic_write_text(output_path, str(result.pop(output_key)))
            artifacts[output_key] = str(output_path)
        artifacts.update(
            {str(k): str(v) for k, v in dict(result.pop("artifacts", {}) or {}).items()}
        )
        payload.update(result)
        return StageResult(
            status=status,
            message=message,
            next_node=result.pop("next_node", None),
            payload=payload,
            artifacts=artifacts,
        )
