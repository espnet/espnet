"""Select and export the best trial."""

from __future__ import annotations

import csv

from espnet3.bin.atlas.core.models import StageResult
from espnet3.bin.atlas.core.utils import save_yaml
from espnet3.bin.atlas.stages.base import BaseStage


class SelectBestStage(BaseStage):
    """Write best-trial artifacts and a sorted leaderboard."""

    study_lock_name = "study_controller"

    def run(self, context) -> StageResult:
        """Write ``leaderboard.csv`` and, if a trial has a score, the best-trial files.

        Args:
            context: The node run's ``StageContext``.

        Returns:
            StageResult: ``status="success"``, with the leaderboard path as
            an artifact.

        Example:
            >>> result = SelectBestStage().run(stage_context)
            >>> result.status
            'success'
        """
        mode = str(context.config.atlas.metric.mode)
        best = context.state.get_best_trial(
            context.scheduler.study_id,
            metric=str(context.config.atlas.metric.name),
            mode=mode,
        )
        trials = [
            trial
            for trial in context.state.list_trials(context.scheduler.study_id)
            if trial.score is not None
        ]
        reverse = mode == "max"
        trials.sort(key=lambda trial: float(trial.score), reverse=reverse)
        leaderboard = context.study_dir / "leaderboard.csv"
        with open(leaderboard, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "rank",
                    "trial_id",
                    "score",
                    "score_name",
                    "status",
                    "decision",
                ],
            )
            writer.writeheader()
            for rank, trial in enumerate(trials, start=1):
                writer.writerow(
                    {
                        "rank": rank,
                        "trial_id": trial.trial_id,
                        "score": trial.score,
                        "score_name": trial.score_name,
                        "status": trial.status,
                        "decision": trial.decision or "",
                    }
                )
        if best is not None:
            save_yaml(
                context.study_dir / "best_trial.yaml",
                {
                    "trial_id": best.trial_id,
                    "score": best.score,
                    "score_name": best.score_name,
                    "config_patch": best.config_patch,
                },
            )
            if (
                best.resolved_config_path is not None
                and best.resolved_config_path.exists()
            ):
                (context.study_dir / "best_config.yaml").write_text(
                    best.resolved_config_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
        return StageResult(
            status="success",
            message="Best trial selected",
            artifacts={"leaderboard": str(leaderboard)},
        )
