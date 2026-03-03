"""Logging helpers for espnet3 experiments."""

from __future__ import annotations

import contextvars
import logging
import os
import shlex
import socket
import subprocess
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from shutil import which
from typing import Mapping

import torch

LOG_FORMAT = (
    "[%(hostname)s] %(asctime)s (%(filename)s:%(lineno)d) "
    "%(levelname)s:\t[%(stage)s] %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S %Z"

# =============================================================================
# Logging Record Setup
# =============================================================================
_LOG_STAGE = contextvars.ContextVar("espnet3_log_stage", default="main")
_BASE_RECORD_FACTORY = logging.getLogRecordFactory()


def _build_record(*args, **kwargs):
    # Inject custom fields used by LOG_FORMAT (stage/hostname) into each LogRecord.
    record = _BASE_RECORD_FACTORY(*args, **kwargs)
    record.stage = _LOG_STAGE.get()
    record.hostname = socket.gethostname()
    return record


logging.setLogRecordFactory(_build_record)


@contextmanager
def log_stage(name: str):
    """Temporarily set the logging stage label used in log records."""
    token = _LOG_STAGE.set(name)
    try:
        yield
    finally:
        _LOG_STAGE.reset(token)


def set_log_format(
    log_format: str | None = None,
    date_format: str | None = None,
    apply: bool = True,
) -> None:
    """Override global log format/date format (optionally update live handlers)."""
    global LOG_FORMAT, DATE_FORMAT
    if log_format is not None:
        LOG_FORMAT = log_format
    if date_format is not None:
        DATE_FORMAT = date_format
    if apply:
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
        root = logging.getLogger()
        for handler in root.handlers:
            handler.setFormatter(formatter)


# =============================================================================
# Logging Configuration (Handlers/Formatters)
# =============================================================================
def _get_next_rotated_log_path(target: Path) -> Path:
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
    log_dir: Path | None = None,
    level: int = logging.INFO,
    filename: str = "run.log",
) -> logging.Logger:
    """Configure logging for an ESPnet3 run.

    This sets up:
      - A root logger with a stream handler (console).
      - An optional file handler at `log_dir/filename`.
      - If `log_dir/filename` already exists, it is rotated to the next
        available suffix (e.g., `run1.log`) and a fresh `run.log` is created.
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
        [babel-t9-28] 2026-02-11 03:57:16 EST (logging_utils.py:376) INFO: [main] hello
        ```

    Example directory tree (when `log_dir/filename` already exists):
        ```
        exp/run1/
        ├── run.log          # new logs (current run)
        └── run1.log         # rotated older logs
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
                rotated = _get_next_rotated_log_path(target)
                os.replace(target, rotated)
            file_handler = logging.FileHandler(target)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    logging.captureWarnings(True)
    return logging.getLogger("espnet3")


def set_stage_log_handler(
    log_dir: Path | None,
    filename: str,
) -> Path | None:
    """Attach a file handler for a stage log, replacing any prior stage handler.

    This function adds a new file handler to the root logger and removes any
    previously installed stage handler (identified via ``_espnet3_stage_log``).
    If a log file already exists at the target path, it is rotated (e.g.,
    ``train.log`` -> ``train1.log``) before creating the new handler.

    Args:
        logger (logging.Logger): Logger used to emit logs.
            The handler is attached to the root logger so the logger hierarchy
            continues to work as expected.
        log_dir (Path | None): Directory that should receive the stage log.
            If None, no handler is installed and None is returned.
        filename (str): Log filename to create within ``log_dir``.

    Returns:
        Path | None: Resolved log file path when installed, otherwise None.
    """
    if log_dir is None:
        return None

    log_dir.mkdir(parents=True, exist_ok=True)
    target = (log_dir / filename).resolve()

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_espnet3_stage_log", False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    if target.exists():
        rotated = _get_next_rotated_log_path(target)
        os.replace(target, rotated)

    file_handler = logging.FileHandler(target)
    file_handler.setFormatter(formatter)
    file_handler._espnet3_stage_log = True
    root.addHandler(file_handler)

    return target


# =============================================================================
# Run Metadata (Command/Git/Requirements)
# =============================================================================
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
    status = _run_git_command(["git", "status", "--short"], cwd)

    dirty = "clean"
    if status is None:
        dirty = "unknown"
    elif status:
        dirty = "dirty"

    meta = {
        "commit": _run_git_command(["git", "rev-parse", "HEAD"], cwd),
        "short_commit": _run_git_command(["git", "rev-parse", "--short", "HEAD"], cwd),
        "branch": _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd),
    }
    meta["worktree"] = dirty
    return meta


def _run_pip_freeze() -> str:
    """Return dependency snapshot output."""
    if which("uv") is not None:
        completed = subprocess.run(
            ["uv", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


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
        logger.log(
            logging.WARNING,
            "Skipping requirements export: no file logger configured.",
            stacklevel=2,
        )
        return

    requirements = _run_pip_freeze()

    target = log_dir / "requirements.txt"
    target.write_text(requirements + "\n", encoding="utf-8")
    logger.log(
        logging.INFO,
        "Wrote requirements snapshot: %s",
        target,
        stacklevel=2,
    )


def log_run_metadata(
    logger: logging.Logger,
    argv: Iterable[str] | None = None,
    configs: Mapping[str, Path | None] | None = None,
    write_requirements: bool = False,
) -> None:
    """Log runtime metadata for the current run.

    Logged fields include:
      - Start timestamp.
      - Python executable and command-line arguments.
      - Working directory (current process directory).
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
        configs={"train": Path("conf/train.yaml")},
        )
        ```

    Example log output (wrapped for readability):
        ```
        [hostname] 2026-02-11 03:57:16 EST (logging_utils.py:376) INFO: [train] \
            === ESPnet3 run started: 2026-02-11T03:57:16.826337 ===
        [hostname] 2026-02-11 03:57:16 EST (run.py:244) INFO: [train] \
            === ESPnet3 run started: 2026-02-11T03:57:16.826430 ===
        [hostname] 2026-02-11 03:57:16 EST (run.py:244) INFO: [train] \
            Command: /path/to/espnet3/tools/.venv/bin/python run.py ...
        [hostname] 2026-02-11 03:57:16 EST (run.py:244) INFO: [train] \
            Python: 3.10.18 (main, Aug 18 2025, 19:18:25) [Clang 20.1.4 ]
        [hostname] 2026-02-11 03:57:16 EST (run.py:244) INFO: [train] \
            Working directory: /path/to/espnet3/egs3/librispeech_100/asr
        [hostname] 2026-02-11 03:57:16 EST (run.py:244) INFO:	[train] \
            train config: /path/to/espnet3/egs3/librispeech_100/asr/conf/train.yaml
        [hostname] 2026-02-11 03:57:16 EST (run.py:244) INFO:	[train] \
            infer config: /path/to/espnet3/egs3/librispeech_100/asr/conf/inference.yaml
        [hostname] 2026-02-11 03:57:16 EST (run.py:244) INFO:	[train] \
            measure config: /path/to/espnet3/egs3/librispeech_100/asr/conf/measure.yaml
        [hostname] 2026-02-11 03:57:17 EST (run.py:244) INFO:	[train] \
            Git: commit=..., short_commit=..., branch=master, worktree=clean
        ```

    Args:
        logger (logging.Logger): Logger used to emit metadata.
        argv (Iterable[str] | None): Command arguments; defaults to sys.argv.
        configs (Mapping[str, Path | None] | None): Named config paths to log.
        write_requirements (bool): If True, export pip freeze output to
            requirements.txt alongside the log file.
    """
    logger.info("=== ESPnet3 run started: %s ===", datetime.now().isoformat())
    logger.log(
        logging.INFO,
        "=== ESPnet3 run started: %s ===",
        datetime.now().isoformat(),
        stacklevel=2,
    )
    cmd_argv = list(argv) if argv is not None else sys.argv
    cmd_text = " ".join(shlex.quote(str(a)) for a in cmd_argv)
    logger.log(
        logging.INFO,
        "Command: %s %s",
        sys.executable,
        cmd_text,
        stacklevel=2,
    )
    logger.log(
        logging.INFO,
        "Python: %s",
        sys.version.replace("\n", " "),
        stacklevel=2,
    )

    cwd = Path.cwd()
    logger.log(logging.INFO, "Working directory: %s", cwd, stacklevel=2)

    if configs:
        for name, path in configs.items():
            if path is None:
                continue
            logger.log(
                logging.INFO,
                "%s config: %s",
                name,
                Path(path).resolve(),
                stacklevel=2,
            )

    git_info = get_git_metadata(cwd)
    if git_info:
        git_parts = [f"{k}={v}" for k, v in git_info.items()]
        logger.log(logging.INFO, "Git: %s", ", ".join(git_parts), stacklevel=2)

    if write_requirements:
        _write_requirements_snapshot(logger)


# =============================================================================
# Environment Metadata (Cluster/Runtime/Torch)
# =============================================================================
def _collect_env(
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
    cluster_prefixes: Iterable[str] | None = None,
    runtime_prefixes: Iterable[str] | None = None,
    runtime_keys: Iterable[str] | None = None,
) -> None:
    """Log selected cluster and runtime environment variables.

    The output includes two blocks:
      - Cluster environment variables (scheduler/runtime IDs).
      - Runtime environment variables (CUDA/NCCL/OMP/PATH, etc.).

    Environment variables collected by default:

    Cluster prefixes:
    | Prefix    | Purpose                                              |
    |-----------|------------------------------------------------------|
    | `SLURM_`  | Slurm job/step metadata (job id, task id, node info) |
    | `PBS_`    | PBS/Torque job metadata                              |
    | `LSF_`    | LSF job metadata                                     |
    | `SGE_`    | SGE job metadata                                     |
    | `COBALT_` | Cobalt job metadata                                  |
    | `OMPI_`   | Open MPI runtime metadata                            |
    | `PMI_`    | PMI (Process Management Interface) metadata          |
    | `MPI_`    | MPI runtime metadata (generic prefix)                |

    Runtime prefixes:
    | Prefix      | Purpose                             |
    |-------------|-------------------------------------|
    | `NCCL_`     | NCCL configuration (multi-GPU comms)|
    | `CUDA_`     | CUDA runtime configuration          |
    | `ROCM_`     | ROCm runtime configuration          |
    | `OMP_`      | OpenMP threading configuration      |
    | `MKL_`      | Intel MKL configuration             |
    | `OPENBLAS_` | OpenBLAS configuration              |
    | `UCX_`      | UCX communication configuration     |
    | `NVIDIA_`   | NVIDIA runtime configuration        |

    Explicit runtime keys:
    | Key                   | Purpose                          |
    |-----------------------|----------------------------------|
    | `PATH`                | Executable search path           |
    | `PYTHONPATH`          | Python module search path        |
    | `LD_LIBRARY_PATH`     | Shared library search path       |
    | `CUDA_VISIBLE_DEVICES`| GPU visibility mask              |
    | `RANK`                | Global rank (distributed)        |
    | `LOCAL_RANK`          | Local rank on node               |
    | `NODE_RANK`           | Node rank in job                 |
    | `WORLD_SIZE`          | Total process count              |
    | `MASTER_ADDR`         | Distributed master address       |
    | `MASTER_PORT`         | Distributed master port          |

    Example usage:
        ```python
        from pathlib import Path
        from espnet3.utils.logging_utils import configure_logging, log_env_metadata

        logger = configure_logging(log_dir=Path("exp/run1"))
        log_env_metadata(logger)
        ```

    Example log output:
        ```
        [hostname] 2026-02-11 03:57:17 EST (run.py:256) INFO:	[train] Cluster env:
            SLURM_JOB_ID=6335268
        [hostname] 2026-02-11 03:57:17 EST (run.py:256) INFO:	[train] Runtime env:
            CUDA_VISIBLE_DEVICES=0
            NCCL_DEBUG=INFO
            PATH=/usr/local/bin:/usr/bin:...
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
    logger.log(logging.INFO, "Cluster env:\n%s", cluster_dump, stacklevel=2)
    logger.log(logging.INFO, "Runtime env:\n%s", runtime_dump, stacklevel=2)

    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None

    logger.log(
        logging.INFO,
        "PyTorch: version=%s, cuda.available=%s, cudnn.version=%s, "
        "cudnn.benchmark=%s, cudnn.deterministic=%s",
        getattr(torch, "__version__", "unknown"),
        torch.cuda.is_available(),
        cudnn_version,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        stacklevel=2,
    )


# =============================================================================
# Introspection Helpers
# =============================================================================
def build_qualified_name(obj) -> str:
    """Return a compact, fully-qualified name for objects or classes.

    Description:
        Produces a stable, human-readable identifier for logging and debugging.
        For objects, it prefers the object's class path. For builtins without a
        module path, it falls back to a truncated string, and includes length
        when available.

    Args:
        obj: Any object or class.

    Returns:
        str: A fully-qualified name or a compact string representation.

    Notes:
        - Builtin types (e.g., list, dict) return "ListClass(len=...)" when
          possible, otherwise a truncated string.
        - For classes, the class module and name are returned.

    Examples:
        ```python
        from pathlib import Path
        build_qualified_name(Path("/tmp"))
        # => 'pathlib.PosixPath'

        build_qualified_name(Path)
        # => 'pathlib.Path'

        build_qualified_name([1, 2, 3])
        # => 'list(len=3)'
        ```
    """
    cls = obj if isinstance(obj, type) else type(obj)
    if not isinstance(obj, type) and cls.__module__ == "builtins":
        if hasattr(obj, "__len__"):
            try:
                return f"{cls.__name__}(len={len(obj)})"
            except Exception:
                pass
        return _truncate_text(str(obj))
    return f"{cls.__module__}.{cls.__name__}"


def build_callable_name(func) -> str:
    """Return a fully-qualified name for callables when possible.

    Description:
        Uses module + qualname for callables (functions, methods, classes with
        __call__). Falls back to build_qualified_name for non-standard callables.

    Args:
        func: Callable object or any object that may be callable.

    Returns:
        str: A fully-qualified callable name if available, else a fallback name.

    Notes:
        - For functions and methods, uses __module__ + __qualname__.
        - For callable instances without __qualname__, falls back to
          build_qualified_name.

    Examples:
        ```python
        def my_fn(x): ...
        build_callable_name(my_fn)
        # => 'my_module.my_fn'

        class MyClass:
            def __call__(self, x): ...
        build_callable_name(MyClass())
        # => 'my_module.MyClass'

        build_callable_name(len)
        # => 'builtins.len'
        ```
    """
    if hasattr(func, "__qualname__") and hasattr(func, "__module__"):
        return f"{func.__module__}.{func.__qualname__}"
    return build_qualified_name(func)


def _iter_attrs(obj) -> Iterable[tuple[str, object]]:
    if not hasattr(obj, "__dict__"):
        return []
    return sorted(
        ((k, v) for k, v in obj.__dict__.items() if not k.startswith("_")),
        key=lambda kv: kv[0],
    )


def _truncate_text(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _dump_attrs(
    logger: logging.Logger,
    obj,
    indent: str,
    depth: int,
    max_depth: int,
    seen: set[int],
) -> None:
    """Log public attributes for an object with bounded recursion depth."""
    if depth > max_depth:
        logger.log(logging.INFO, "%s...", indent, stacklevel=3)
        return
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    for key, value in _iter_attrs(obj):
        if isinstance(value, torch.utils.data.Dataset):
            logger.log(
                logging.INFO,
                "%s%s: %r",
                indent,
                key,
                value,
                stacklevel=3,
            )
            continue
        if isinstance(value, torch.nn.Module):
            logger.log(
                logging.INFO,
                "%s%s: %s",
                indent,
                key,
                build_qualified_name(value),
                stacklevel=3,
            )
            continue
        if isinstance(value, Iterator):
            logger.log(
                logging.INFO,
                "%s%s: %s",
                indent,
                key,
                _truncate_text(str(value)),
                stacklevel=3,
            )
            continue
        if value is None:
            summary = "None"
        elif isinstance(value, (str, bytes, int, float, bool)):
            summary = repr(value)
        elif isinstance(value, Path):
            summary = repr(str(value))
        else:
            summary = None

        if summary is not None:
            logger.log(
                logging.INFO,
                "%s%s: %s",
                indent,
                key,
                summary,
                stacklevel=3,
            )
            continue

        logger.log(
            logging.INFO,
            "%s%s: %s",
            indent,
            key,
            build_qualified_name(value),
            stacklevel=3,
        )
        _dump_attrs(
            logger,
            value,
            indent=indent * 2,
            depth=depth + 1,
            max_depth=max_depth,
            seen=seen,
        )


def log_component(
    logger: logging.Logger,
    kind: str,
    label: str,
    obj,
    max_depth: int,
) -> None:
    """Log a component instance with class info, repr, and attributes.

    Description:
        Emits a structured log block for a single object. The block includes
        a class line, a representation line, and a recursive attribute dump
        up to the specified depth.

    Args:
        logger (logging.Logger): Logger used to emit messages.
        kind (str): Label prefix for the entry (e.g., "Component", "Env").
        label (str): Human-readable label identifying the entry.
        obj: Object instance to log. If None, the function returns early.
        max_depth (int): Maximum depth for recursive attribute dumping.

    Raises:
        None

    Returns:
        None

    Example:
        ```python
        from espnet3.utils.logging_utils import log_component

        # Custom class instance.
        class CustomThing:
            def __init__(self, name: str, value: int):
                self.name = name
                self.value = value

        log_component(logger, "Custom", "example", CustomThing("demo", 7))
        ```

        Example log output:
        ```
        Custom[example] class: my_module.CustomThing
        Custom[example]: <my_module.CustomThing object at ...>
          name: 'demo'
          value: 7
        ```

    Notes:
        - The logger uses `stacklevel=2` so log lines point at the caller.
        - Attribute dumping uses `build_qualified_name` for readable class names.
        - Set `max_depth` to 0 to log only the class and repr lines.
    """
    if obj is None:
        return
    logger.log(
        logging.INFO,
        "%s[%s] class: %s",
        kind,
        label,
        build_qualified_name(obj),
        stacklevel=2,
    )
    logger.log(logging.INFO, "%s[%s]: %r", kind, label, obj, stacklevel=2)
    _dump_attrs(
        logger,
        obj,
        indent="  ",
        depth=0,
        max_depth=max_depth,
        seen=set(),
    )


def log_instance_dict(
    logger: logging.Logger,
    kind: str,
    entries: dict[str, object],
    max_depth: int = 2,
) -> None:
    """Log a dictionary of instances with a common kind label.

    Description:
        Iterates over the provided mapping and logs each value using the shared
        `kind` label. This is useful for dumping collections of structured
        objects (e.g., environment info blocks or component registries) in a
        consistent, readable format.

    Args:
        logger (logging.Logger): Logger used to emit messages.
        kind (str): Label prefix for each entry (e.g., "Env", "Component").
        entries (dict[str, object]): Mapping from entry keys to objects.
        max_depth (int): Maximum depth for attribute dumping.

    Returns:
        None

    Notes:
        - Empty mappings are ignored.
        - Each entry is logged via `log_component`, which emits a class line,
          a repr line, and selected public attributes.

    Examples:
        ```python
        log_instance_dict(
            logger,
            kind="Env",
            entries={"CUDA": torch.cuda, "NCCL": nccl_module},
        )
        ```

    Raises:
        None
    """
    if not entries:
        return
    for key, value in entries.items():
        log_component(
            logger,
            kind=kind,
            label=str(key),
            obj=value,
            max_depth=max_depth,
        )
