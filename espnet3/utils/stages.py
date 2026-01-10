"""Helpers for selecting and running stage methods on system objects."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence

logger = logging.getLogger(__name__)


def resolve_stages(
    requested: Sequence[str],
    stages: Sequence[str],
) -> List[str]:
    """Resolve a requested stage list against the available stages.

    The special token ``"all"`` expands to all known stages in order.
    Otherwise, the returned list preserves the order of ``stages`` and
    includes only those present in ``requested``.
    """
    if "all" in requested:
        return list(stages)
    requested_set = set(requested)
    return [s for s in stages if s in requested_set]


def run_stages(
    system: Any,
    stages_to_run: Iterable[str],
    *,
    dry_run: bool = False,
    log: logging.Logger | None = None,
    stage_log_dir_fn: Callable[[str], Path | None] | None = None,
    on_stage_start: Callable[[str, logging.Logger], None] | None = None,
) -> None:
    """Invoke stage methods on ``system`` in order with logging and timing.

    Args:
        system: Object providing stage methods named in ``stages_to_run``.
        stages_to_run: Iterable of stage method names to execute.
        dry_run: If True, log intended stages without executing them.
        log: Optional logger instance; defaults to module logger.
        stage_log_dir_fn: Optional callable that returns a stage log directory.
            When provided (or resolved from ``system.get_stage_log_dir``),
            a per-stage file handler is configured using
            ``<log_dir>/<stage>.log`` before executing each stage.
        on_stage_start: Optional hook invoked after stage logging is configured.
            This can be used to emit per-stage metadata (configs, environment,
            requirements snapshots, etc.) into the newly attached log file.

    Raises:
        AttributeError: If a named stage method is missing on ``system``.
        TypeError: If a stage method rejects CLI-provided arguments.
        Exception: Re-raises any exception from a stage method.
    """
    log = log or logger
    if stage_log_dir_fn is None and hasattr(system, "get_stage_log_dir"):
        stage_log_dir_fn = getattr(system, "get_stage_log_dir")
    for stage in stages_to_run:
        fn = getattr(system, stage, None)
        if fn is None:
            raise AttributeError(f"System has no stage method: {stage}")

        if dry_run:
            log.info("[DRY RUN] would run stage: %s", stage)
            continue

        if stage_log_dir_fn is not None:
            from espnet3.utils.logging import set_stage_log_handler

            log_dir = stage_log_dir_fn(stage)
            set_stage_log_handler(
                log,
                Path(log_dir) if log_dir else None,
                filename=f"{stage}.log",
            )
            if on_stage_start is not None:
                on_stage_start(stage, log)

        start = time.perf_counter()
        log.info("=== [START] stage: %s ===", stage)
        try:
            fn()
        except TypeError as e:
            log.exception("Stage '%s' failed (bad arguments)", stage)
            raise TypeError(
                f"Stage '{stage}' does not accept CLI arguments; "
                "put all settings in the YAML config."
            ) from e
        except Exception:
            elapsed = time.perf_counter() - start
            log.exception("Stage '%s' failed after %.2fs", stage, elapsed)
            raise
        else:
            elapsed = time.perf_counter() - start
            log.info("=== [DONE] stage: %s (%.2fs) ===", stage, elapsed)
