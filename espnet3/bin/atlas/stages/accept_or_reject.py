"""Accept or reject a trial relative to the current best."""

from __future__ import annotations

from espnet3.bin.atlas.core.models import StageResult
from espnet3.bin.atlas.stages.base import BaseStage


class AcceptOrRejectStage(BaseStage):
    """Compare the latest score with the best known score."""

    study_lock_name = "study_controller"

    def run(self, context) -> StageResult:
        """Accept the trial as the new best, or reject it, and write ``decision.md``.

        The first trial, or one improving on the current best (per
        ``metric.mode``), is accepted and recorded as the study's best;
        otherwise it is rejected. Either way updates the trial's
        ``status``/``decision`` and writes a short decision summary.

        Args:
            context: The node run's ``StageContext``; must have a current
                trial with a ``score`` already set.

        Returns:
            StageResult: ``status="accepted"`` or ``status="rejected"``.

        Example:
            >>> result = AcceptOrRejectStage().run(stage_context)
            >>> result.status
            'accepted'
        """
        trial = context.current_trial
        assert trial is not None
        mode = str(context.config.atlas.metric.mode)
        best = context.state.get_best_trial(
            context.scheduler.study_id,
            metric=str(context.config.atlas.metric.name),
            mode=mode,
        )
        previous = context.state.get_best_trial(
            context.scheduler.study_id,
            metric=str(context.config.atlas.metric.name),
            mode=mode,
            exclude_trial_id=trial.trial_id,
        )
        accepted = False
        if best is None or best.trial_id == trial.trial_id:
            accepted = True
        elif mode == "min":
            accepted = float(trial.score) < float(best.score)
        else:
            accepted = float(trial.score) > float(best.score)

        decision_path = context.get_trial_dir() / "decision.md"
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
                    f"- best_before: {previous.score if previous else 'n/a'}",
                    "",
                    message,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        context.state.update_trial(trial)
        return StageResult(
            status=status, message=message, artifacts={"decision": str(decision_path)}
        )
