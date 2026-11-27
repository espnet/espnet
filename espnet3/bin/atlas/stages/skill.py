"""Generic stage that runs a graph node's skill through the coding agent."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

from espnet3.bin.atlas.core.models import AgentRequest, StageResult, Trial
from espnet3.bin.atlas.core.utils import (
    read_optional,
    to_plain_data,
    write_text_atomically,
)
from espnet3.bin.atlas.stages.base import BaseStage


def build_reply_schema(node) -> dict:
    """Build the JSON Schema of a skill node's reply.

    The reply always has a ``status`` (one of the node's ``on`` keys, or
    ``terminal`` for a node with none) and a ``message``, plus the fields the
    graph declares under the node's ``output_schema``. Every field is required
    and no other field is allowed, as constrained decoding needs.
    """
    properties = {
        "status": {"type": "string", "enum": list(node.on) or ["terminal"]},
        "message": {"type": "string", "description": "short result description"},
        **node.output_schema,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


class SkillStage(BaseStage):
    """Run a graph node's selected skill through the configured coding agent.

    Builds an ``AgentRequest`` from the study's objective, knowledge pack,
    trial history, and the node's ``skill_context``, dispatches it to
    ``context.agent``, applies any ``trial_updates`` the agent returns to the
    current trial, and writes the node's ``output_file`` when one is
    configured.
    """

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

        trial = context.current_trial
        trial_dir = context.get_trial_dir() if trial else context.get_artifact_dir()
        allowed_actions = list(node.allowed_actions or ["read_only"])
        code_edit_config = getattr(context.config.atlas, "code_edit", None)
        if node.skill == "atlas_hyperparameter_search_implement_code" and not bool(
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
            repo_context="",
            trial_history_csv=read_optional(trials_csv),
            latest_metrics=trial.metrics if trial else {},
            context_files={
                "study_dir": str(context.study_dir),
                "recipe_dir": str(context.recipe_dir),
                "trial_dir": str(trial_dir),
                "current_trial": json.dumps(asdict(trial), indent=2) if trial else "",
                "skill_context": json.dumps(node.skill_context, indent=2),
                "code_edit_config": json.dumps(
                    to_plain_data(getattr(context.config.atlas, "code_edit", {})),
                    indent=2,
                ),
                "stop_after_trial": str(context.scheduler.should_stop()).lower(),
            },
            allowed_actions=allowed_actions,
            output_schema=build_reply_schema(node),
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
                write_text_atomically(output_path, node.fallback_output)
                return StageResult(
                    status="terminal",
                    message="Wrote fallback output after skill agent failure",
                    artifacts={"output": str(output_path)},
                )
            return StageResult(
                status="failure", message=response.message or "Skill agent failed"
            )

        result = dict(response.structured or {})
        status = result.pop("status")
        message = result.pop("message")
        trial_updates = {key: result[key] for key in node.trial_update_keys}
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
        if node.output_file and output_key:
            assert output_path is not None
            text = str(result.pop(output_key))
            if node.output_append and output_path.is_file():
                existing = output_path.read_text(encoding="utf-8").rstrip("\n")
                text = f"{existing}\n\n{text.strip(chr(10))}\n"
            write_text_atomically(output_path, text)
            artifacts[output_key] = str(output_path)
        return StageResult(
            status=status, message=message, payload=result, artifacts=artifacts
        )
