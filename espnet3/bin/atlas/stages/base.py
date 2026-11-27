"""Atlas stage base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from filelock import FileLock

from espnet3.bin.atlas.core.models import StageResult
from espnet3.bin.atlas.core.utils import get_utc_now, write_text_atomically


class BaseStage(ABC):
    """Base class implementing idempotent, optionally lock-guarded stage execution.

    Subclasses implement :meth:`run`. :meth:`execute` wraps it with
    success/failure marker files under the node run's artifact directory so a
    re-dispatched node run short-circuits to the cached result instead of
    re-running, and with an optional study-wide file lock (``study_lock_name``)
    so only one worker executes a locked stage (such as ``SkillStage``'s
    agent call) at a time.
    """

    study_lock_name: str | None = None
    """Name of the study-wide lock file to hold while running, or ``None``
    to run without a lock."""

    def execute(self, context) -> StageResult:
        """Run this stage once, idempotently, caching the result on disk.

        Args:
            context: The node run's ``StageContext`` (study/graph/trial state).

        Returns:
            StageResult: The cached result from a prior successful run, or
            the fresh result from calling :meth:`run`.

        Example:
            >>> result = SkillStage().execute(stage_context)
            >>> result.status
            'success'
        """
        marker_dir = context.get_artifact_dir()
        marker_dir.mkdir(parents=True, exist_ok=True)
        success_path = marker_dir / ".stage_success.json"
        failure_path = marker_dir / ".stage_failure.json"
        started_path = marker_dir / ".stage_started.json"
        if success_path.exists() and not bool(context.config.get("force", False)):
            return StageResult.from_json(success_path.read_text(encoding="utf-8"))
        write_text_atomically(started_path, f'{{"started_at": "{get_utc_now()}"}}')
        if self.study_lock_name is None:
            result = self.run(context)
        else:
            with FileLock(context.study_dir / f".lock_{self.study_lock_name}"):
                result = self.run(context)
        target = success_path
        if result.status in {"failure", "timeout", "retry"}:
            target = failure_path
        write_text_atomically(target, result.to_json())
        return result

    @abstractmethod
    def run(self, context) -> StageResult:
        """Run this stage's logic once; subclasses must implement it.

        Args:
            context: The node run's ``StageContext`` (study/graph/trial state).

        Returns:
            StageResult: The stage's status, message, and any payload or
            artifacts.

        Example:
            >>> result = SkillStage().run(stage_context)
            >>> result.status
            'success'
        """
