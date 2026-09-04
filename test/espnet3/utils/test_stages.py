import argparse
import contextlib
import logging

import pytest

from espnet3.systems.base.system import BaseSystem
from espnet3.utils import stages_utils
from espnet3.utils.stages_utils import (
    _RANK_ENV_KEYS,
    parse_cli_and_stage_args,
    resolve_stages,
    run_stages,
)


class DummySystem:
    def __init__(self):
        self.calls = []
        self.stage_log_dirs = {"default": None}

    def stage_a(self):
        self.calls.append("a")

    def stage_b(self):
        self.calls.append("b")


def test_resolve_stages_all():
    stages = ["stage_a", "stage_b"]
    assert resolve_stages(["all"], stages) == stages


def test_resolve_stages_subset_preserves_stage_order():
    stages = ["stage_a", "stage_b", "stage_c"]
    assert resolve_stages(["stage_b", "stage_a"], stages) == ["stage_a", "stage_b"]


def test_parse_cli_and_stage_args_resolves_requested_stage_order(monkeypatch):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=["all"])
    monkeypatch.setattr(
        parser,
        "parse_args",
        lambda: argparse.Namespace(stages=["stage_b", "stage_a"]),
    )

    args, stages_to_run = parse_cli_and_stage_args(parser, ["stage_a", "stage_b"])

    assert args.stages == ["stage_b", "stage_a"]
    assert stages_to_run == ["stage_a", "stage_b"]


def test_get_process_rank_reads_rank_from_environment(monkeypatch):
    for key in _RANK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RANK", "7")

    assert stages_utils._get_process_rank() == 7


def test_run_stages_dry_run_skips_execution(caplog):
    system = DummySystem()
    with caplog.at_level(logging.INFO):
        run_stages(
            system,
            ["stage_a", "stage_b"],
            args=argparse.Namespace(dry_run=True),
        )

    assert system.calls == []
    assert "[DRY RUN] would run stage: stage_a" in caplog.text
    assert "[DRY RUN] would run stage: stage_b" in caplog.text


def test_run_stages_missing_method_raises():
    system = DummySystem()
    with pytest.raises(AttributeError, match="System has no stage method: stage_c"):
        run_stages(system, ["stage_c"])


def test_run_stages_typeerror_wrapped():
    class BadSystem(BaseSystem):
        def stage_a(self, arg):
            return arg

    system = BadSystem()
    with pytest.raises(
        TypeError,
        match="Stage 'stage_a' does not accept CLI arguments",
    ):
        run_stages(system, ["stage_a"])


def test_run_stages_reraises_exception(monkeypatch):
    class CrashSystem(BaseSystem):
        def stage_a(self):
            raise ValueError("boom")

    system = CrashSystem()
    monkeypatch.setattr(stages_utils, "log_stage", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(stages_utils, "log_stage_metadata", lambda *_, **__: None)
    monkeypatch.setattr(stages_utils, "set_stage_log_handler", lambda *_, **__: None)
    with pytest.raises(ValueError, match="boom"):
        run_stages(system, ["stage_a"])


def test_run_stages_writes_stage_logs(tmp_path):
    class LoggingSystem:
        def __init__(self, log_dir):
            self.log_dir = log_dir
            self.stage_log_dirs = {"default": log_dir}

        def stage_a(self):
            pass

        def stage_b(self):
            pass

    system = LoggingSystem(tmp_path)
    logger = logging.getLogger("espnet3.test_stages")
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)
    try:
        run_stages(system, ["stage_a", "stage_b"], log=logger)
    finally:
        root.setLevel(previous_level)

    stage_a_log = tmp_path / "stage_a.log"
    stage_b_log = tmp_path / "stage_b.log"
    assert stage_a_log.exists()
    assert stage_b_log.exists()

    stage_a_text = stage_a_log.read_text(encoding="utf-8")
    stage_b_text = stage_b_log.read_text(encoding="utf-8")
    assert "stage: stage_a" in stage_a_text
    assert "stage: stage_b" not in stage_a_text
    assert "stage: stage_b" in stage_b_text
    assert "stage: stage_a" not in stage_b_text


def test_run_stages_train_uses_per_rank_log_filename(monkeypatch, tmp_path):
    seen = {}

    class TrainingSystem:
        def __init__(self):
            self.stage_log_dirs = {"default": tmp_path, "train": tmp_path}
            self.training_config = argparse.Namespace(stage_log_mode="per_rank")

        def train(self):
            seen["called"] = True

    monkeypatch.setattr(stages_utils, "log_stage", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(stages_utils, "log_stage_metadata", lambda *_, **__: None)
    monkeypatch.setattr(stages_utils, "_get_process_rank", lambda: 3)

    def fake_set_stage_log_handler(log_dir, filename):
        seen["log_dir"] = log_dir
        seen["filename"] = filename

    monkeypatch.setattr(
        stages_utils, "set_stage_log_handler", fake_set_stage_log_handler
    )

    run_stages(TrainingSystem(), ["train"])

    assert seen["called"] is True
    assert seen["log_dir"] == tmp_path
    assert seen["filename"] == "train_rank3.log"


def test_run_stages_train_unknown_log_mode_falls_back_to_rank0(
    monkeypatch, tmp_path, caplog
):
    seen = {}

    class TrainingSystem:
        def __init__(self):
            self.stage_log_dirs = {"default": tmp_path, "train": tmp_path}
            self.training_config = argparse.Namespace(stage_log_mode="unknown")

        def train(self):
            seen["called"] = True

    monkeypatch.setattr(stages_utils, "log_stage", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(stages_utils, "log_stage_metadata", lambda *_, **__: None)
    monkeypatch.setattr(stages_utils, "_get_process_rank", lambda: 0)

    def fake_set_stage_log_handler(log_dir, filename):
        seen["log_dir"] = log_dir
        seen["filename"] = filename

    monkeypatch.setattr(
        stages_utils, "set_stage_log_handler", fake_set_stage_log_handler
    )

    with caplog.at_level(logging.ERROR):
        run_stages(TrainingSystem(), ["train"])

    assert seen["called"] is True
    assert seen["log_dir"] == tmp_path
    assert seen["filename"] == "train.log"
    assert "Unknown stage_log_mode='unknown'" in caplog.text
