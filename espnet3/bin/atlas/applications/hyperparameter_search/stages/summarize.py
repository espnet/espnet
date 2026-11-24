"""Write the final study summary."""

from __future__ import annotations

from espnet3.bin.atlas.core.models import StageResult
from espnet3.bin.atlas.core.reports import write_summary
from espnet3.bin.atlas.core.stage import AtlasStage


class SummarizeStage(AtlasStage):
    """Generate summary.md from current study state."""

    study_lock_name = "study_controller"

    def run(self, context) -> StageResult:
        """Write ``summary.md`` from the study's trials and best trial, ending the graph.

        Args:
            context: The node run's ``StageContext``.

        Returns:
            StageResult: ``status="terminal"`` (ends the graph), with the
            summary path as an artifact.

        Example:
            >>> result = SummarizeStage().run(stage_context)
            >>> result.status
            'terminal'
        """
        trials = context.state.list_trials(context.scheduler.study_id)
        best = context.state.get_best_trial(
            context.scheduler.study_id,
            metric=str(context.config.atlas.metric.name),
            mode=str(context.config.atlas.metric.mode),
        )
        objective = (context.study_dir / "program.md").read_text(encoding="utf-8")
        path = write_summary(context.study_dir, objective, trials, best)
        return StageResult(
            status="terminal",
            message="Summary generated",
            artifacts={"summary": str(path)},
        )
