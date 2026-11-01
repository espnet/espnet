"""Tests for ESPnet3 stage machinery."""

import argparse
import contextlib
import logging
from typing import ClassVar

import pytest
from omegaconf import OmegaConf

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem
from espnet3.utils import stages_utils
from espnet3.utils.stages_utils import _RANK_ENV_KEYS, resolve_stages, run_stages


@pytest.fixture
def no_bake(monkeypatch):
    """Fixture to disable baking during tests."""
    monkeypatch.setattr(stages_utils, "bake", lambda *a, **k: None)


class DummySystem:
    """A dummy system for testing stage execution."""

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="stage_a", config="training"),
        StageSpec(name="stage_b", config="training"),
    )
    exp_dir = None
    stage_configs = {"stage_a": OmegaConf.create({}), "stage_b": OmegaConf.create({})}
    own_keys = {"stage_a": (), "stage_b": ()}

    def __init__(self):
        """Initialize the dummy system."""
        self.calls = []

    def stage_a(self):
        """Run stage a."""
        self.calls.append("a")

    def stage_b(self):
        """Run stage b."""
        self.calls.append("b")


def test_resolve_stages_all():
    """Test that resolve_stages works for all stages."""
    stages = ["stage_a", "stage_b"]
    assert resolve_stages(["all"], stages) == stages


def test_resolve_stages_subset_preserves_stage_order():
    """Test that resolve_stages preserves the order of a subset."""
    stages = ["stage_a", "stage_b", "stage_c"]
    assert resolve_stages(["stage_b", "stage_a"], stages) == ["stage_a", "stage_b"]


def test_get_process_rank_reads_rank_from_environment(monkeypatch):
    """Test that the process rank is correctly read from environment variables."""
    for key in _RANK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RANK", "7")

    assert stages_utils._get_process_rank() == 7


def test_run_stages_dry_run_skips_execution(caplog, no_bake):
    """Test that a dry run skips actual stage execution."""
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
    """Test that a missing stage method raises an AttributeError."""
    with pytest.raises(AttributeError, match="System has no stage method: stage_c"):
        run_stages(DummySystem(), ["stage_c"])


def test_run_stages_typeerror_wrapped(no_bake, tmp_path):
    """Test that type errors in stage arguments are wrapped."""

    class BadSystem(BaseSystem):
        stages: ClassVar[tuple[StageSpec, ...]] = (
            StageSpec(name="stage_a", config="training"),
        )

        def __init__(self, configs=None, exp_dir=None):
            super().__init__(configs=configs, exp_dir=exp_dir)

        def stage_a(self, arg):
            return arg

    system = BadSystem(configs={"training": OmegaConf.create({})}, exp_dir=tmp_path)
    with pytest.raises(
        TypeError,
        match="Stage 'stage_a' does not accept CLI arguments",
    ):
        run_stages(system, ["stage_a"])


def test_run_stages_reraises_exception(monkeypatch, tmp_path, no_bake):
    """Test that exceptions in stages are re-raised."""

    class CrashSystem(BaseSystem):
        stages: ClassVar[tuple[StageSpec, ...]] = (
            StageSpec(name="stage_a", config="training"),
        )

        def __init__(self, configs=None, exp_dir=None):
            super().__init__(configs=configs, exp_dir=exp_dir)

        def stage_a(self):
            raise ValueError("boom")

    system = CrashSystem(configs={"training": OmegaConf.create({})}, exp_dir=tmp_path)
    monkeypatch.setattr(stages_utils, "log_stage", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(stages_utils, "log_stage_metadata", lambda *_, **__: None)
    monkeypatch.setattr(stages_utils, "set_stage_log_handler", lambda *_, **__: None)
    with pytest.raises(ValueError, match="boom"):
        run_stages(system, ["stage_a"])


def test_run_stages_writes_stage_logs(tmp_path, no_bake):
    """Test that stage logs are correctly written."""

    class LoggingSystem:
        stages: ClassVar[tuple[StageSpec, ...]] = (
            StageSpec(name="stage_a", config="training"),
            StageSpec(name="stage_b", config="training"),
        )

        def __init__(self, log_dir):
            self.exp_dir = log_dir
            self.stage_configs = {
                "stage_a": OmegaConf.create({}),
                "stage_b": OmegaConf.create({}),
            }
            self.own_keys = {"stage_a": (), "stage_b": ()}

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


def test_run_stages_train_uses_per_rank_log_filename(monkeypatch, tmp_path, no_bake):
    """Test that training stages use rank-specific log filenames."""
    seen = {}

    class TrainingSystem:
        stages: ClassVar[tuple[StageSpec, ...]] = (
            StageSpec(name="train", config="training"),
        )

        def __init__(self, exp_dir=None, stage_configs=None, own_keys=None):
            self.exp_dir = exp_dir
            self.stage_configs = stage_configs
            self.own_keys = own_keys

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

    system = TrainingSystem(
        exp_dir=tmp_path,
        stage_configs={"train": OmegaConf.create({"stage_log_mode": "per_rank"})},
        own_keys={"train": ()},
    )
    run_stages(system, ["train"])

    assert seen["called"] is True
    assert seen["log_dir"] == tmp_path
    assert seen["filename"] == "train_rank3.log"


def test_run_stages_writes_full_config_snapshot(tmp_path, monkeypatch):
    """Integration: run_stages() actually persists <log_dir>/<stage>.yaml

    via snapshot_config - not just the unit-tested function in isolation.
    """

    class TrainingSystem:
        stages: ClassVar[tuple[StageSpec, ...]] = (
            StageSpec(name="train", config="training", log_dir="exp_dir"),
        )

        def __init__(self, exp_dir=None, stage_configs=None, own_keys=None):
            self.exp_dir = exp_dir
            self.stage_configs = stage_configs
            self.own_keys = own_keys

        def train(self):
            pass

    monkeypatch.setattr(stages_utils, "_get_process_rank", lambda: 0)

    system = TrainingSystem(
        exp_dir=tmp_path,
        stage_configs={
            "train": OmegaConf.create(
                {"exp_dir": str(tmp_path), "seed": 42, "model": "dummy"}
            )
        },
        own_keys={"train": ()},
    )
    run_stages(system, ["train"])

    snapshot_path = tmp_path / "train.yaml"
    assert snapshot_path.is_file()
    content = snapshot_path.read_text(encoding="utf-8")
    assert "seed: 42" in content
    assert "model: dummy" in content


def test_run_stages_bakes_each_stage_config_before_running(monkeypatch):
    """Test that each stage is baked before running."""
    calls = []

    def fake_bake(exp_dir, stage, config, *, own_keys):
        calls.append((stage, own_keys))

    monkeypatch.setattr(stages_utils, "bake", fake_bake)
    monkeypatch.setattr(stages_utils, "_get_process_rank", lambda: 0)

    system = DummySystem()
    run_stages(system, ["stage_a", "stage_b"])

    assert calls == [("stage_a", ()), ("stage_b", ())]
    assert system.calls == ["a", "b"]


def test_run_stages_dry_run_and_other_ranks_do_not_bake(monkeypatch):
    """Test that dry runs and other ranks do not trigger baking."""
    calls = []

    def fake_bake(exp_dir, stage, config, *, own_keys):
        calls.append((stage, own_keys))

    monkeypatch.setattr(stages_utils, "bake", fake_bake)

    system = DummySystem()
    # dry run -> calls == [] and system.calls == []
    monkeypatch.setattr(stages_utils, "_get_process_rank", lambda: 0)
    run_stages(system, ["stage_a", "stage_b"], args=argparse.Namespace(dry_run=True))
    assert calls == []
    assert system.calls == []

    # then rank 2 (_get_process_rank -> 2),
    # real run -> calls == [], system.calls == ["a", "b"]
    monkeypatch.setattr(stages_utils, "_get_process_rank", lambda: 2)
    run_stages(system, ["stage_a", "stage_b"])
    assert calls == []
    assert system.calls == ["a", "b"]


def test_run_stages_train_unknown_log_mode_falls_back_to_rank0(
    monkeypatch, tmp_path, caplog
):
    """Test that unknown log mode falls back to rank 0 logging."""
    seen = {}

    class TrainingSystem:
        stages: ClassVar[tuple[StageSpec, ...]] = (
            StageSpec(name="train", config="training"),
        )

        def __init__(self, exp_dir=None, stage_configs=None, own_keys=None):
            self.exp_dir = exp_dir
            self.stage_configs = stage_configs
            self.own_keys = own_keys

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

    system = TrainingSystem(
        exp_dir=tmp_path,
        stage_configs={"train": OmegaConf.create({"stage_log_mode": "unknown"})},
        own_keys={"train": ()},
    )
    with caplog.at_level(logging.ERROR):
        run_stages(system, ["train"])

    assert seen["called"] is True
    assert seen["log_dir"] == tmp_path
    assert seen["filename"] == "train.log"
    assert "Unknown stage_log_mode='unknown'" in caplog.text
