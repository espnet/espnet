"""Append trial results to the shared knowledge pack."""

from __future__ import annotations

import json
from pathlib import Path

from espnet3.bin.atlas.core.models import StageResult
from espnet3.bin.atlas.core.stage import AtlasStage


class UpdateKnowledgeStage(AtlasStage):
    """Append a trial's patch + outcome to knowledge_pack.md."""

    def run(self, context) -> StageResult:
        """Append the current trial's patch and outcome to ``knowledge_pack.md``.

        Writes an "Early Stopping Record" section (with a do-not-reuse
        lesson) for an early-stopped trial, or a normal trial record
        (status, score, patch, rationale, reflection) otherwise, excluding
        runtime-only patch keys (``recipe_dir``, ``exp_dir``, etc.) from the
        displayed patch. No-ops if there is no current trial or no
        ``knowledge_pack.md`` yet.

        Args:
            context: The node run's ``StageContext``.

        Returns:
            StageResult: ``status="done"`` if the study should now stop,
            ``status="continue"`` otherwise, or ``status="success"`` for the
            no-op cases.

        Example:
            >>> result = UpdateKnowledgeStage().run(stage_context)
            >>> result.status
            'continue'
        """
        trial = context.current_trial
        if trial is None:
            return StageResult(status="success", message="No trial to record")

        knowledge_path = context.study_dir / "knowledge" / "knowledge_pack.md"
        if not knowledge_path.exists():
            return StageResult(status="success", message="No knowledge_pack.md found")

        patch = trial.config_patch or {}
        _runtime_keys = {"recipe_dir", "data_dir", "exp_tag", "exp_dir", "stats_dir"}
        display_patch = {k: v for k, v in patch.items() if k not in _runtime_keys}

        if trial.status == "early_stopped":
            lines = [
                "",
                f"## Early Stopping Record: {trial.trial_id}",
                "",
                f"**Config patch (DO NOT reuse this direction):**",
                f"```json",
                json.dumps(display_patch, ensure_ascii=False, indent=2),
                f"```",
                "",
            ]
            if trial.rationale:
                lines += [f"**Original rationale:** {trial.rationale.splitlines()[0]}", ""]
            if getattr(trial, "reflection", None):
                lines += ["**Early stop reason:**", trial.reflection.strip(), ""]
            lines.append(
                f"**Lesson:** This config was stopped early due to poor training trajectory. "
                f"Do not propose similar configurations."
            )
        else:
            lines = [
                "",
                f"## Trial {trial.trial_id}",
                f"- **status**: {trial.status}",
                f"- **score** ({trial.score_name}): {trial.score}",
                f"- **config_patch**: `{json.dumps(display_patch, ensure_ascii=False)}`",
            ]
            if trial.rationale:
                lines.append(f"- **rationale**: {trial.rationale.splitlines()[0]}")
            if trial.expected_effect:
                lines.append(f"- **expected_effect**: {trial.expected_effect.splitlines()[0]}")
            if getattr(trial, "reflection", None):
                lines += ["", "**Reflection and comparison evidence:**", trial.reflection.strip(), ""]
            if getattr(trial, "next_suggestion", None):
                lines.append(f"- **next_suggestion**: {trial.next_suggestion.splitlines()[0]}")

        existing = knowledge_path.read_text(encoding="utf-8")
        knowledge_path.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")

        if context.scheduler.should_stop():
            return StageResult(
                status="done",
                message=f"Updated knowledge_pack.md with {trial.trial_id}, stopping",
            )
        return StageResult(
            status="continue",
            message=f"Updated knowledge_pack.md with {trial.trial_id}",
        )
