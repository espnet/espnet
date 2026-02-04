import logging

import pytest

from espnet3.utils.stages_utils import resolve_stages, run_stages


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
