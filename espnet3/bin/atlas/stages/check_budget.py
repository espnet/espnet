"""Decide whether the study continues after a trial has been recorded."""

from __future__ import annotations

from espnet3.bin.atlas.core.models import StageResult
from espnet3.bin.atlas.stages.base import BaseStage


class CheckBudgetStage(BaseStage):
    """Route to ``done`` when the study's budget is spent, else ``continue``."""

    def run(self, context) -> StageResult:
        """Ask the scheduler whether the study should stop now.

        Args:
            context: The node run's ``StageContext``.

        Returns:
            StageResult: ``status="done"`` if ``scheduler.should_stop()`` is true,
            ``status="continue"`` otherwise.

        Example:
            >>> CheckBudgetStage().run(stage_context).status
            'continue'
        """
        if context.scheduler.should_stop():
            return StageResult(status="done", message="Budget spent, stopping")
        return StageResult(status="continue", message="Budget remains")
