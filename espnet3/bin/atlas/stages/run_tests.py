"""Run unit tests written by write_files and loop back on failure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from espnet3.bin.atlas.core.models import StageResult
from espnet3.bin.atlas.core.utils import get_utc_now, save_yaml
from espnet3.bin.atlas.stages.base import BaseStage


def _create_next_run_dir(trial_dir: Path) -> tuple[Path, int]:
    """Return (new_run_dir, run_index_1based) under trial_dir/test_runs/."""
    base = trial_dir / "test_runs"
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.iterdir())
    n = len(existing) + 1
    d = base / f"run_{n:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d, n


def _find_test_dirs(recipe_dir: Path, allowed_dirs: list[str]) -> list[Path]:
    """Return directories (or files) that pytest should collect from."""
    candidates: list[Path] = []
    for rel in allowed_dirs:
        d = recipe_dir / rel
        if not d.exists():
            continue
        # Prefer an explicit tests/ subdir; fall back to the whole allowed dir.
        tests_subdir = d / "tests"
        if tests_subdir.is_dir():
            candidates.append(tests_subdir)
        else:
            # Look for test files directly inside the allowed dir (not recursive
            # into subdirs already captured above).
            for p in d.rglob("test_*.py"):
                if "tests" not in p.parent.name:
                    candidates.append(p)
    return candidates if candidates else [recipe_dir / rel for rel in allowed_dirs]


class RunTestsStage(BaseStage):
    """Execute pytest over test files written by ``write_files``.

    On success routes to ``run_trial``.
    On failure saves the error log to ``trial_dir/test_runs/run_NNN/error.log``
    (picked up by the next ``write_files`` invocation) and routes back.
    After ``code_edit.max_test_retries`` consecutive failures routes to
    ``max_retries_exceeded`` so the graph can bail out to ``reflect``.
    """

    def run(self, context) -> StageResult:
        """Run pytest over the trial's ``allowed_dirs`` and route on the outcome.

        Args:
            context: The node run's ``StageContext``; must have a current
                trial.

        Returns:
            StageResult: ``status="success"`` on a pytest pass or no test
            files found, ``status="failure"`` on a pytest failure (routes
            back to ``write_files``), or ``status="max_retries_exceeded"``
            once ``code_edit.max_test_retries`` consecutive runs have
            failed.

        Example:
            >>> result = RunTestsStage().run(stage_context)
            >>> result.status
            'success'
        """
        trial = context.current_trial
        assert trial is not None

        code_edit_cfg = getattr(context.config.atlas, "code_edit", None)
        allowed_dirs: list[str] = list(getattr(code_edit_cfg, "allowed_dirs", []) or [])
        max_retries: int = int(getattr(code_edit_cfg, "max_test_retries", 5))

        trial_dir = context.get_trial_dir()
        run_dir, run_index = _create_next_run_dir(trial_dir)

        if run_index > max_retries:
            return StageResult(
                status="max_retries_exceeded",
                message=f"Tests failed {run_index - 1} times — giving up",
                payload={"trial_id": trial.trial_id, "run_index": run_index},
            )

        test_targets = _find_test_dirs(context.recipe_dir, allowed_dirs)
        if not test_targets:
            return StageResult(
                status="success",
                message="No test files found — skipping run_tests",
                payload={"trial_id": trial.trial_id},
            )

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--tb=short",
            "-x",  # stop on first failure
            "--no-header",
            "-o",
            "addopts=",  # Ignore repo-wide coverage flags; no coverage plugin needed.
            "-q",
            *[str(t) for t in test_targets],
        ]

        log_path = run_dir / "pytest.log"
        proc = subprocess.run(
            cmd,
            cwd=str(context.recipe_dir),
            capture_output=True,
            text=True,
        )
        combined = proc.stdout + ("\n" + proc.stderr if proc.stderr.strip() else "")
        log_path.write_text(combined, encoding="utf-8")

        save_yaml(
            run_dir / "meta.yaml",
            {
                "trial_id": trial.trial_id,
                "run_index": run_index,
                "returncode": proc.returncode,
                "cmd": cmd,
                "timestamp": get_utc_now(),
            },
        )

        if proc.returncode == 0:
            return StageResult(
                status="success",
                message=f"All tests passed (run {run_index})",
                payload={"trial_id": trial.trial_id, "run_index": run_index},
                artifacts={"pytest_log": str(log_path)},
            )

        # Tests failed — write error.log so write_files can pick it up
        error_log = run_dir / "error.log"
        error_log.write_text(combined, encoding="utf-8")

        context.logger.warning(
            "run_tests: attempt %d/%d failed — routing back to write_files",
            run_index,
            max_retries,
        )

        return StageResult(
            status="failure",
            message=f"Tests failed (run {run_index}/{max_retries}): see {log_path}",
            payload={"trial_id": trial.trial_id, "run_index": run_index},
            artifacts={"pytest_log": str(log_path), "error_log": str(error_log)},
        )
