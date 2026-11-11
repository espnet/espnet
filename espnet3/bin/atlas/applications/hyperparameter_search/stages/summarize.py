"""Write the final study summary."""

from __future__ import annotations

from espnet3.bin.atlas.core.models import StageResult
from espnet3.bin.atlas.core.stage import AtlasStage
from espnet3.bin.atlas.reports.summary import write_summary


class SummarizeStage(AtlasStage):
    """Generate summary.md from current study state."""

    study_lock_name = "study_controller"

    def run(self, context) -> StageResult:
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
