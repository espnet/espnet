"""Helpers for selecting and running stage methods on system objects."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from espnet3.utils.logging_utils import (
    log_stage,
    log_stage_metadata,
    set_stage_log_handler,
)

logger = logging.getLogger(__name__)

_RANK_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "SLURM_PROCID",
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "MPI_RANK",
)


def _get_process_rank() -> int:
    """Return current process rank from torch.distributed or env vars."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass

    for key in _RANK_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None and value.isdigit():
            return int(value)
    return 0


def _get_stage_log_mode(system: Any) -> str:
    """Return normalized stage_log_mode from training_config, defaulting to rank0."""
    mode = "rank0"
    training_config = getattr(system, "training_config", None)
    if training_config is not None:
        mode = getattr(training_config, "stage_log_mode", mode)
    return str(mode).lower()


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


def parse_cli_and_stage_args(
    parser: argparse.ArgumentParser,
    stages: Sequence[str],
) -> Tuple[argparse.Namespace, List[str]]:
    """Parse CLI arguments and expand the requested stage selection.

    This helper is a thin wrapper around ``ArgumentParser.parse_args()`` plus
    :func:`resolve_stages`. It keeps the runner entrypoints concise and ensures
    the `"all"` shorthand is expanded consistently across recipes.

    Args:
        parser (argparse.ArgumentParser): Parser configured by the recipe
            entrypoint. It is expected to define a ``--stages`` argument whose
            value is compatible with ``resolve_stages(...)``.
        stages (Sequence[str]): Ordered list of stage names supported by the
            current runner, for example ``["create_dataset", "train", "infer"]``.

    Returns:
        Tuple[argparse.Namespace, List[str]]: A pair containing:
            - the parsed CLI namespace returned by ``parser.parse_args()``
            - the resolved list of stages to execute, preserving the canonical
              order from ``stages``

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument("--stages", nargs="+", default=["all"])
        >>> # If argv is: ["--stages", "infer", "train"]
        >>> args, stages_to_run = parse_cli_and_stage_args(
        ...     parser,
        ...     ["create_dataset", "train", "infer"],
        ... )
        >>> stages_to_run
        ['train', 'infer']

    Notes:
        The returned stage order is not the same as CLI input order when they
        differ. The canonical order defined by ``stages`` always wins so stage
        execution remains deterministic.
    """
    args = parser.parse_args()
    stages_to_run = resolve_stages(args.stages, stages)
    return args, stages_to_run


def run_stages(
    system: Any,
    stages_to_run: Iterable[str],
    args: argparse.Namespace | None = None,
    log: logging.Logger | None = None,
) -> None:
    """Invoke stage methods on ``system`` in order with logging and timing.

    Args:
        system: Object providing stage methods named in ``stages_to_run``.
        stages_to_run: Iterable of stage method names to execute.
        args: Parsed CLI arguments. ``dry_run`` and ``write_requirements`` are
            read from here when present.
        log: Optional logger instance; defaults to module logger.

    Raises:
        AttributeError: If a named stage method is missing on ``system``.
        TypeError: If a stage method rejects CLI-provided arguments.
        Exception: Re-raises any exception from a stage method.
    """
    log = log or logger
    dry_run = bool(getattr(args, "dry_run", False))
    for stage in stages_to_run:
        fn = getattr(system, stage, None)
        if fn is None:
            raise AttributeError(f"System has no stage method: {stage}")

        with log_stage(stage):
            if dry_run:
                log.info("[DRY RUN] would run stage: %s", stage)
                continue

            stage_log_dirs = system.stage_log_dirs
            log_dir = stage_log_dirs.get(stage) or stage_log_dirs.get("default")
            filename = f"{stage}.log"

            if stage == "train":
                # stage_log_mode controls per-rank logging: "rank0" or "per_rank".
                # rank0 avoids multi-process rotation races;
                # per_rank writes per-rank logs.
                stage_log_mode = _get_stage_log_mode(system)
                rank = _get_process_rank()
                if stage_log_mode not in {"rank0", "per_rank"}:
                    log.error(
                        "Unknown stage_log_mode=%r (expected 'rank0' or 'per_rank'); "
                        "falling back to 'rank0'.",
                        stage_log_mode,
                    )
                    stage_log_mode = "rank0"
                if stage_log_mode == "rank0" and rank != 0:
                    # Non-zero ranks skip file logging in rank0 mode.
                    log_dir = None

                filename = (
                    f"{stage}.log"
                    if stage_log_mode == "rank0"
                    else f"{stage}_rank{rank}.log"
                )

            set_stage_log_handler(
                Path(log_dir) if log_dir else None,
                filename=filename,
            )
            log_stage_metadata(log, system=system, args=args)

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
