"""Logging helpers for espnet3 experiments."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from shutil import which
from typing import Iterable, Mapping

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _next_rotated_log_path(target: Path) -> Path:
    """Return the next available rotated log path (e.g., run1.log)."""
    suffixes = target.suffixes
    suffix = "".join(suffixes)
    base = target.name[: -len(suffix)] if suffix else target.name

    index = 1
    while True:
        candidate = target.with_name(f"{base}{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def configure_logging(
    *,
    log_dir: Path | None = None,
    level: int = logging.INFO,
    filename: str = "run.log",
) -> logging.Logger:
    """Configure logging for an ESPnet3 run.

    This sets up:
      - A root logger with a stream handler (console).
      - An optional file handler at `log_dir/filename`.
      - Warning capture into the logging system.

    Example usage:
        ```python
        from pathlib import Path
        from espnet3.utils.logging_utils import configure_logging

        logger = configure_logging(log_dir=Path("exp/run1"), level=logging.INFO)
        logger.info("hello")
        ```

    Example log output:
        ```
        2026-02-04 10:15:22 | INFO | espnet3 | hello
        ```

    Args:
        log_dir (Path | None): Directory to store the log file.
            If None, only console logging is configured.
        level (int): Logging level (e.g., logging.INFO).
        filename (str): Log file name when `log_dir` is provided.

    Returns:
        logging.Logger: Logger instance named "espnet3".
    """
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        target = (log_dir / filename).resolve()
        has_file = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None)
            and Path(h.baseFilename).resolve() == target
            for h in root.handlers
        )
        if not has_file:
            if target.exists():
                rotated = _next_rotated_log_path(target)
                os.replace(target, rotated)
            file_handler = logging.FileHandler(target)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    logging.captureWarnings(True)
    return logging.getLogger("espnet3")


def _run_git_command(cmd: list[str], cwd: Path | None) -> str | None:
    """Run a git command and return stdout, or None on failure."""
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def get_git_metadata(cwd: Path | None = None) -> dict[str, str]:
    """Return git metadata for the current repository.

    This attempts to read commit hash, short hash, branch name, and worktree
    status from the git repository rooted at `cwd`.

    Args:
        cwd (Path | None): Directory within the target git repo.

    Returns:
        dict[str, str]: Collected metadata keys, possibly including:
            - "commit": Full commit hash.
            - "short_commit": Abbreviated commit hash.
            - "branch": Current branch name.
            - "worktree": "clean", "dirty", or "unknown".
    """
    cwd = cwd or Path.cwd()
    head = _run_git_command(["git", "rev-parse", "HEAD"], cwd)
    branch = _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    short = _run_git_command(["git", "rev-parse", "--short", "HEAD"], cwd)
    status = _run_git_command(["git", "status", "--short"], cwd)

    dirty = "clean"
    if status is None:
        dirty = "unknown"
    elif status:
        dirty = "dirty"

    meta: dict[str, str] = {}
    if head:
        meta["commit"] = head
    if short:
        meta["short_commit"] = short
    if branch:
        meta["branch"] = branch
    meta["worktree"] = dirty
    return meta


def format_command(argv: Iterable[str] | None = None) -> str:
    """Format command arguments into a shell-escaped string."""
    argv = list(argv) if argv is not None else sys.argv
    return " ".join(shlex.quote(str(a)) for a in argv)


def _run_pip_freeze() -> str | None:
    """Return `pip freeze` output, or None on failure."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        if which("uv") is None:
            return None
        try:
            completed = subprocess.run(
                ["uv", "pip", "freeze"],
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return None


def _get_log_dir_from_logger(logger: logging.Logger) -> Path | None:
    """Return the log directory from any configured file handler."""
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(
            handler, "baseFilename", None
        ):
            return Path(handler.baseFilename).resolve().parent

    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(
            handler, "baseFilename", None
        ):
            return Path(handler.baseFilename).resolve().parent

    return None


def _write_requirements_snapshot(logger: logging.Logger) -> None:
    """Write a requirements snapshot alongside the configured log file."""
    log_dir = _get_log_dir_from_logger(logger)
    if log_dir is None:
        logger.warning("Skipping requirements export: no file logger configured.")
        return

    requirements = _run_pip_freeze()
    if requirements is None:
        logger.warning("Failed to export requirements via pip freeze.")
        return

    target = log_dir / "requirements.txt"
    target.write_text(requirements + "\n", encoding="utf-8")
    logger.info("Wrote requirements snapshot: %s", target)


def log_run_metadata(
    logger: logging.Logger,
    *,
    argv: Iterable[str] | None = None,
    workdir: Path | None = None,
    configs: Mapping[str, Path | None] | None = None,
    write_requirements: bool = False,
) -> None:
    """Log runtime metadata for the current run.

    Logged fields include:
      - Start timestamp.
      - Python executable and command-line arguments.
      - Working directory.
      - Python version.
      - Config paths (if provided).
      - Git metadata (commit/branch/dirty), when available.
      - Optional requirements snapshot (pip freeze).

    Example usage:
        ```python
        from pathlib import Path
        from espnet3.utils.logging_utils import configure_logging, log_run_metadata

        logger = configure_logging(log_dir=Path("exp/run1"))
        log_run_metadata(
            logger,
            argv=["espnet3-train", "--config", "conf/train.yaml"],
            workdir=Path("/home/user/espnet3"),
            configs={"train": Path("conf/train.yaml")},
        )
        ```

    Example log output (wrapped for readability, <= 88 chars):
        ```
        2026-02-04 10:15:22 | INFO | espnet3 | === ESPnet3 run started: \
            2026-02-04T10:15:22 ===
        2026-02-04 10:15:22 | INFO | espnet3 | Command: /usr/bin/python3 \
            espnet3-train --config conf/train.yaml
        2026-02-04 10:15:22 | INFO | espnet3 | Python: 3.10.12 (GCC 11.4.0)
        2026-02-04 10:15:22 | INFO | espnet3 | Working directory: /home/user/espnet3
        2026-02-04 10:15:22 | INFO | espnet3 | train config: /home/user/espnet3/conf/\
            train.yaml
        2026-02-04 10:15:22 | INFO | espnet3 | Git: commit=0123456789abcdef, \
            short_commit=0123456, branch=main, worktree=clean
        ```

    Args:
        logger (logging.Logger): Logger used to emit metadata.
        argv (Iterable[str] | None): Command arguments; defaults to sys.argv.
        workdir (Path | None): Working directory to report.
        configs (Mapping[str, Path | None] | None): Named config paths to log.
        write_requirements (bool): If True, export pip freeze output to
            requirements.txt alongside the log file.
    """
    logger.info("=== ESPnet3 run started: %s ===", datetime.now().isoformat())
    logger.info("Command: %s %s", sys.executable, format_command(argv))
    logger.info("Python: %s", sys.version.replace("\n", " "))

    cwd = workdir or Path.cwd()
    logger.info("Working directory: %s", cwd)

    if configs:
        for name, path in configs.items():
            if path is None:
                continue
            logger.info("%s config: %s", name, Path(path).resolve())

    git_info = get_git_metadata(cwd)
    if git_info:
        git_parts = [f"{k}={v}" for k, v in git_info.items()]
        logger.info("Git: %s", ", ".join(git_parts))

    if write_requirements:
        _write_requirements_snapshot(logger)


def _collect_env(
    *,
    prefixes: Iterable[str] | None = None,
    keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Collect environment variables matching prefixes or explicit keys.

    Args:
        prefixes (Iterable[str] | None): Prefixes to match (e.g., "CUDA_").
        keys (Iterable[str] | None): Exact variable names to include.

    Returns:
        dict[str, str]: Sorted environment variables that match.
    """
    prefixes = tuple(prefixes or ())
    key_set = {k for k in (keys or ())}
    collected: dict[str, str] = {}
    for name, value in os.environ.items():
        if name in key_set or any(name.startswith(prefix) for prefix in prefixes):
            collected[name] = value
    return dict(sorted(collected.items()))


def log_env_metadata(
    logger: logging.Logger,
    *,
    cluster_prefixes: Iterable[str] | None = None,
    runtime_prefixes: Iterable[str] | None = None,
    runtime_keys: Iterable[str] | None = None,
) -> None:
    """Log selected cluster and runtime environment variables.

    The output includes two blocks:
      - Cluster environment variables (scheduler/runtime IDs).
      - Runtime environment variables (CUDA/NCCL/OMP/PATH, etc.).

    Example usage:
        ```python
        from pathlib import Path
        from espnet3.utils.logging_utils import configure_logging, log_env_metadata

        logger = configure_logging(log_dir=Path("exp/run1"))
        log_env_metadata(logger)
        ```

    Example log output:
        ```
        2026-02-04 10:15:22 | INFO | espnet3 | Cluster env:
        SLURM_JOB_ID=12345
        SLURM_PROCID=0
        2026-02-04 10:15:22 | INFO | espnet3 | Runtime env:
        CUDA_VISIBLE_DEVICES=0
        NCCL_DEBUG=INFO
        PATH=/usr/local/bin:/usr/bin:/bin
        ```

    Args:
        logger (logging.Logger): Logger used to emit metadata.
        cluster_prefixes (Iterable[str] | None): Prefixes for cluster variables.
        runtime_prefixes (Iterable[str] | None): Prefixes for runtime variables.
        runtime_keys (Iterable[str] | None): Explicit runtime keys to include.
    """
    cluster_prefixes = cluster_prefixes or (
        "SLURM_",
        "PBS_",
        "LSF_",
        "SGE_",
        "COBALT_",
        "OMPI_",
        "PMI_",
        "MPI_",
    )
    runtime_prefixes = runtime_prefixes or (
        "NCCL_",
        "CUDA_",
        "ROCM_",
        "OMP_",
        "MKL_",
        "OPENBLAS_",
        "UCX_",
        "NVIDIA_",
    )
    runtime_keys = runtime_keys or (
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "RANK",
        "LOCAL_RANK",
        "NODE_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    )

    cluster_env = _collect_env(prefixes=cluster_prefixes)
    runtime_env = _collect_env(prefixes=runtime_prefixes, keys=runtime_keys)

    cluster_dump = "\n".join(f"{k}={v}" for k, v in cluster_env.items()) or "(none)"
    runtime_dump = "\n".join(f"{k}={v}" for k, v in runtime_env.items()) or "(none)"
    logger.info("Cluster env:\n%s", cluster_dump)
    logger.info("Runtime env:\n%s", runtime_dump)
