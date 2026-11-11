"""Accept or reject a trial relative to the current best."""

from __future__ import annotations

from espnet3.bin.atlas.core.models import StageResult
from espnet3.bin.atlas.core.stage import AtlasStage


class AcceptOrRejectStage(AtlasStage):
    """Compare the latest score with the best known score."""

    study_lock_name = "study_controller"

    def run(self, context) -> StageResult:
        trial = context.current_trial
        assert trial is not None
        mode = str(context.config.atlas.metric.mode)
        best = context.state.get_best_trial(
            context.scheduler.study_id,
            metric=str(context.config.atlas.metric.name),
            mode=mode,
        )
        accepted = False
        if best is None or best.trial_id == trial.trial_id:
            accepted = True
        elif mode == "min":
            accepted = float(trial.score) < float(best.score)
        else:
            accepted = float(trial.score) > float(best.score)

        decision_path = context.trial_dir() / "decision.md"
        if accepted:
            trial.status = "accepted"
            trial.decision = "accepted"
            context.state.set_best_trial(context.scheduler.study_id, trial.trial_id)
            message = "Improved current best score."
            status = "accepted"
        else:
            trial.status = "rejected"
            trial.decision = "rejected"
            message = "Did not improve current best score."
            status = "rejected"
        decision_path.write_text(
            "\n".join(
                [
                    f"# Decision: {trial.decision}",
                    "",
                    f"- trial: {trial.trial_id}",
                    f"- score: {trial.score}",
                    f"- best_before: {best.score if best and best.trial_id != trial.trial_id else 'n/a'}",
                    "",
                    message,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        context.state.update_trial(trial)
        return StageResult(status=status, message=message, artifacts={"decision": str(decision_path)})
