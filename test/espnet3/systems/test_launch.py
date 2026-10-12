"""Tests for espnet3.systems.launch: build_parser, launch, default_conf_package."""

import logging

import pytest
from omegaconf import OmegaConf

from espnet3.contract.stages import StageContractError, StageSpec, roles
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.launch import build_parser, default_conf_package, launch


class _RecordingSystem(BaseSystem):
    """A minimal system whose stages just record that they ran.

    ``launch()`` builds the instance internally, so tests that need to
    inspect it after the call read it back from ``instances``.
    """

    stages = (
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
    )
    instances: list = []

    def __init__(self, **kwargs):
        self.calls = []
        self._default_log_dir = None
        type(self).instances.append(self)

    def train(self):
        self.calls.append("train")

    def infer(self):
        self.calls.append("infer")


@pytest.fixture
def stub_config_loading(monkeypatch):
    """Replace config loading and experiment-context wiring with no-ops.

    ``launch()``'s own orchestration (stage resolution, contract checks,
    system instantiation, delegating to ``run_stages``) is what these tests
    exercise; loading real packaged config files and validating experiment
    context are independently tested elsewhere.
    """
    import espnet3.systems.launch as launch_module

    captured = {}

    def fake_load_and_merge_config(
        config_path, config_name, default_package=None, **kwargs
    ):
        captured.setdefault("default_package", default_package)
        return OmegaConf.create({})

    monkeypatch.setattr(
        launch_module, "load_and_merge_config", fake_load_and_merge_config
    )
    monkeypatch.setattr(
        launch_module, "configure_logging", lambda *a, **k: logging.getLogger("test")
    )
    monkeypatch.setattr(
        launch_module, "apply_training_experiment_context", lambda *a, **k: None
    )
    monkeypatch.setattr(
        launch_module, "validate_experiment_context", lambda *a, **k: None
    )
    monkeypatch.setattr(launch_module, "resolve_loaded_configs", lambda *a, **k: None)
    return captured


def test_build_parser_stages_choices_follow_declared_order():
    parser = build_parser(_RecordingSystem)
    action = next(a for a in parser._actions if a.dest == "stages")
    assert action.choices == ["train", "infer", "all"]


def test_build_parser_has_one_config_arg_per_declared_role():
    parser = build_parser(_RecordingSystem)
    args = parser.parse_args(["--training_config", "a.yaml"])
    for role in roles(_RecordingSystem):
        assert hasattr(args, f"{role}_config")
    assert str(args.training_config) == "a.yaml"


def test_build_parser_has_no_config_arg_for_an_undeclared_role():
    parser = build_parser(_RecordingSystem)
    assert "metrics" not in roles(_RecordingSystem)
    with pytest.raises(SystemExit):
        parser.parse_args(["--metrics_config", "b.yaml"])


def test_build_parser_rejects_unknown_stage_name():
    parser = build_parser(_RecordingSystem)
    with pytest.raises(SystemExit):
        parser.parse_args(["--stages", "decode"])


def test_launch_requires_config_for_requested_stage(stub_config_loading):
    with pytest.raises(StageContractError, match="train runs on the training config"):
        launch(_RecordingSystem, argv=["--stages", "train"])


def test_launch_dry_run_does_not_run_stages(stub_config_loading, tmp_path):
    _RecordingSystem.instances.clear()
    launch(
        _RecordingSystem,
        argv=[
            "--stages",
            "train",
            "--training_config",
            str(tmp_path / "training.yaml"),
            "--dry_run",
        ],
    )
    assert _RecordingSystem.instances[-1].calls == []


def test_default_conf_package_derives_egs3_template_path():
    from espnet3.systems.esp2_asr.system import ASRSystem

    assert default_conf_package(ASRSystem) == "egs3.TEMPLATE.esp2_asr"


def test_launch_conf_package_overrides_the_derived_default(
    stub_config_loading, tmp_path
):
    launch(
        _RecordingSystem,
        conf_package="some.other.package",
        argv=[
            "--stages",
            "train",
            "--training_config",
            str(tmp_path / "training.yaml"),
        ],
    )
    assert stub_config_loading["default_package"] == "some.other.package"
