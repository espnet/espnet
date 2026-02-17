from __future__ import annotations

import io
import logging as py_logging
from pathlib import Path

from espnet3.utils import logging_utils as elog


def _reset_logger(logger: py_logging.Logger, handlers, level, propagate: bool) -> None:
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_configure_logging_adds_console_and_file(tmp_path: Path):
    root = py_logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    old_propagate = root.propagate
    try:
        root.handlers = []
        root.propagate = False

        logger = elog.configure_logging(log_dir=tmp_path, filename="run.log")
        logger.info("hello")

        file_handlers = [
            h for h in root.handlers if isinstance(h, py_logging.FileHandler)
        ]
        stream_handlers = [
            h for h in root.handlers if isinstance(h, py_logging.StreamHandler)
        ]

        assert (tmp_path / "run.log").exists()
        assert len(file_handlers) == 1
        assert len(stream_handlers) >= 1
    finally:
        _reset_logger(root, old_handlers, old_level, old_propagate)


def test_configure_logging_is_idempotent(tmp_path: Path):
    root = py_logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    old_propagate = root.propagate
    try:
        root.handlers = []
        root.propagate = False

        elog.configure_logging(log_dir=tmp_path, filename="run.log")
        elog.configure_logging(log_dir=tmp_path, filename="run.log")

        file_handlers = [
            h for h in root.handlers if isinstance(h, py_logging.FileHandler)
        ]
        assert len(file_handlers) == 1
    finally:
        _reset_logger(root, old_handlers, old_level, old_propagate)


def test_configure_logging_rotates_existing_file(tmp_path: Path):
    root = py_logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    old_propagate = root.propagate
    try:
        root.handlers = []
        root.propagate = False

        log_path = tmp_path / "run.log"
        log_path.write_text("old log\n", encoding="utf-8")

        elog.configure_logging(log_dir=tmp_path, filename="run.log")

        rotated = tmp_path / "run1.log"
        assert rotated.exists()
        assert rotated.read_text(encoding="utf-8") == "old log\n"
        assert log_path.exists()
    finally:
        _reset_logger(root, old_handlers, old_level, old_propagate)


def test_log_run_metadata_logs_command_and_git(monkeypatch, tmp_path: Path):
    logger = py_logging.getLogger("espnet3.test.logging")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    stream = io.StringIO()
    handler = py_logging.StreamHandler(stream)
    logger.handlers = [handler]
    logger.setLevel(py_logging.INFO)
    logger.propagate = False

    monkeypatch.setattr(
        elog, "get_git_metadata", lambda cwd=None: {"commit": "abc", "branch": "main"}
    )

    try:
        elog.log_run_metadata(
            logger,
            argv=["run.py", "--arg", "1"],
            workdir=tmp_path,
            configs={"train": tmp_path / "train.yaml"},
        )
        out = stream.getvalue()
        assert "run.py --arg 1" in out
        assert str(tmp_path) in out
        assert "train.yaml" in out
        assert "commit=abc" in out and "branch=main" in out
        assert "Python:" in out
    finally:
        _reset_logger(logger, old_handlers, old_level, old_propagate)


def test_log_run_metadata_writes_requirements(monkeypatch, tmp_path: Path):
    logger = py_logging.getLogger("espnet3.test.requirements")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    handler = py_logging.FileHandler(tmp_path / "run.log")
    logger.handlers = [handler]
    logger.setLevel(py_logging.INFO)
    logger.propagate = False

    monkeypatch.setattr(elog, "_run_pip_freeze", lambda: "pkg==1.2.3")

    try:
        elog.log_run_metadata(logger, write_requirements=True)
        contents = (tmp_path / "requirements.txt").read_text(encoding="utf-8")
        assert "pkg==1.2.3" in contents
    finally:
        handler.close()
        _reset_logger(logger, old_handlers, old_level, old_propagate)
