# espnet3/utils/stages.py
from __future__ import annotations

import logging
import time
from typing import Any, Iterable, List, Sequence

logger = logging.getLogger(__name__)


def resolve_stages(
    requested: Sequence[str],
    stages: Sequence[str],
) -> List[str]:
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
) -> None:
    log = log or logger
    for stage in stages_to_run:
        fn = getattr(system, stage, None)
        if fn is None:
            raise AttributeError(f"System has no stage method: {stage}")

        if dry_run:
            log.info("[DRY RUN] would run stage: %s", stage)
            continue

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
