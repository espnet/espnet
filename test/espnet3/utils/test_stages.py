import logging

import pytest

from espnet3.utils.stages import resolve_stages, run_stages


class DummySystem:
    def __init__(self):
        self.calls = []

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


def test_run_stages_dry_run_skips_execution(caplog):
    system = DummySystem()
    with caplog.at_level(logging.INFO):
        run_stages(system, ["stage_a", "stage_b"], dry_run=True)

    assert system.calls == []
    assert "[DRY RUN] would run stage: stage_a" in caplog.text
    assert "[DRY RUN] would run stage: stage_b" in caplog.text


def test_run_stages_missing_method_raises():
    system = DummySystem()
    with pytest.raises(AttributeError, match="System has no stage method: stage_c"):
        run_stages(system, ["stage_c"])


def test_run_stages_typeerror_wrapped():
    class BadSystem:
        def stage_a(self, arg):
            return arg

    system = BadSystem()
    with pytest.raises(
        TypeError,
        match="Stage 'stage_a' does not accept CLI arguments",
    ):
        run_stages(system, ["stage_a"])


def test_run_stages_reraises_exception():
    class CrashSystem:
        def stage_a(self):
            raise ValueError("boom")

    system = CrashSystem()
    with pytest.raises(ValueError, match="boom"):
        run_stages(system, ["stage_a"])


def test_run_stages_writes_stage_logs(tmp_path):
    class LoggingSystem:
        def __init__(self, log_dir):
            self.log_dir = log_dir

        def get_stage_log_dir(self, stage):
            return self.log_dir

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
