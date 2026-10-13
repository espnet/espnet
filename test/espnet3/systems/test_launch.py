"""Tests for espnet3.systems.launch: build_parser, launch, default_conf_package."""

import importlib.util
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from espnet3.contract.stages import StageContractError, StageSpec, roles
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.launch import build_parser, default_conf_package, launch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RECIPE_RUN_PY_FILES = [
    _REPO_ROOT / "egs3" / "mini_an4" / "esp2_asr" / "run.py",
    _REPO_ROOT / "egs3" / "librispeech_100" / "esp2_asr" / "run.py",
    _REPO_ROOT / "egs3" / "spgispeech" / "esp2_asr" / "run.py",
]


def _exec_run_py(path: Path, module_name: str):
    """Load and execute ``path`` as a module named ``module_name``.

    Mirrors how a Dask/DataLoader worker re-imports ``run.py`` under its
    own module name (never ``"__main__"``), as opposed to how a user
    actually invokes it as a script.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    def __init__(self, *, configs, exp_dir=None, stages_to_run=None):
        self.configs = configs
        self.exp_dir = exp_dir
        self.stages_to_run = stages_to_run
        self.calls = []
        type(self).instances.append(self)

    def train(self):
        self.calls.append("train")

    def infer(self):
        self.calls.append("infer")


@pytest.fixture
def stub_config_loading(monkeypatch):
    """Replace config loading with a no-op; record what role it was for.

    ``launch()``'s own orchestration (stage resolution, contract checks,
    system instantiation, delegating to ``run_stages``) is what these tests
    exercise; loading real packaged config files is independently tested
    elsewhere.
    """
    import espnet3.systems.launch as launch_module

    captured = {}

    def fake_load_and_merge_config(
        config_path, config_name, default_package=None, **kwargs
    ):
        captured.setdefault("default_package", default_package)
        captured.setdefault("loaded_for", []).append(config_name)
        return OmegaConf.create({})

    monkeypatch.setattr(
        launch_module, "load_and_merge_config", fake_load_and_merge_config
    )
    monkeypatch.setattr(
        launch_module, "configure_logging", lambda *a, **k: logging.getLogger("test")
    )
    monkeypatch.setattr(launch_module, "run_stages", lambda *a, **k: None)
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


def test_build_parser_has_exp_dir_and_dry_run_args():
    parser = build_parser(_RecordingSystem)
    args = parser.parse_args(["--stages", "train", "--training_config", "a.yaml"])

    assert args.exp_dir is None
    assert args.dry_run is False


def test_build_parser_accepts_exp_dir_and_dry_run():
    parser = build_parser(_RecordingSystem)
    args = parser.parse_args(
        [
            "--stages",
            "train",
            "--training_config",
            "a.yaml",
            "--exp_dir",
            "./exp/my_run",
            "--dry_run",
        ]
    )

    assert args.exp_dir == "./exp/my_run"
    assert args.dry_run is True


def test_launch_passes_configs_and_exp_dir(stub_config_loading, tmp_path):
    _RecordingSystem.instances.clear()

    launch(
        _RecordingSystem,
        argv=[
            "--stages",
            "train",
            "--training_config",
            str(tmp_path / "training.yaml"),
            "--exp_dir",
            str(tmp_path / "exp" / "my_run"),
        ],
    )

    system = _RecordingSystem.instances[-1]
    assert system.exp_dir == str(tmp_path / "exp" / "my_run")
    assert "training" in system.configs
    assert system.configs["training"] is not None


def test_launch_passes_none_for_a_role_with_no_cli_config(
    stub_config_loading, tmp_path
):
    _RecordingSystem.instances.clear()

    launch(
        _RecordingSystem,
        argv=[
            "--stages",
            "train",
            "--training_config",
            str(tmp_path / "training.yaml"),
            "--exp_dir",
            str(tmp_path / "exp" / "my_run"),
        ],
    )

    system = _RecordingSystem.instances[-1]
    assert system.configs.get("inference") is None


def test_launch_requires_config_or_exp_dir_for_requested_stage(stub_config_loading):
    with pytest.raises(StageContractError, match="train runs on the training config"):
        launch(_RecordingSystem, argv=["--stages", "train"])


def test_launch_allows_a_missing_config_when_exp_dir_is_given(
    stub_config_loading, tmp_path
):
    # The stage's config may come from its baked file under --exp_dir;
    # check_requested_stages must not demand a CLI config in that case.
    _RecordingSystem.instances.clear()

    launch(
        _RecordingSystem,
        argv=["--stages", "train", "--exp_dir", str(tmp_path / "exp" / "my_run")],
    )

    assert _RecordingSystem.instances[-1].stages_to_run == ["train"]


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


class _CustomRoleSystem(BaseSystem):
    """A system declaring a custom config role outside the five standard ones."""

    stages = (StageSpec(name="finetune", config="finetuning"),)
    instances: list = []

    def __init__(self, *, configs, exp_dir=None, stages_to_run=None):
        self.configs = configs
        self.exp_dir = exp_dir
        self.stages_to_run = stages_to_run
        type(self).instances.append(self)

    def finetune(self):
        pass


def test_launch_passes_custom_role_config_to_constructor(stub_config_loading, tmp_path):
    """A custom role must reach system_cls(configs=...) keyed by its own name."""
    _CustomRoleSystem.instances.clear()
    launch(
        _CustomRoleSystem,
        argv=[
            "--stages",
            "finetune",
            "--finetuning_config",
            str(tmp_path / "finetuning.yaml"),
        ],
    )
    system = _CustomRoleSystem.instances[-1]
    assert "finetuning" in system.configs
    assert system.configs["finetuning"] is not None


def test_launch_inference_only_stage_runs_without_training_config(
    stub_config_loading, tmp_path
):
    """An inference-only request must not require --training_config."""
    _RecordingSystem.instances.clear()
    launch(
        _RecordingSystem,
        argv=[
            "--stages",
            "infer",
            "--inference_config",
            str(tmp_path / "inference.yaml"),
        ],
    )
    system = _RecordingSystem.instances[-1]
    assert system.calls == ["infer"]


# ---------------------------------------------------------------------------
# Regression: recipe run.py must guard launch() with `if __name__ == "__main__"`
# ---------------------------------------------------------------------------
#
# Without the guard, a Dask (or DataLoader, on macOS with spawn) worker
# re-imports run.py under its own module name and re-runs the whole
# pipeline, racing the parent and locking an already-locked shard.


@pytest.mark.parametrize("run_py", _RECIPE_RUN_PY_FILES, ids=lambda p: p.parent.name)
def test_importing_run_py_does_not_execute_the_pipeline(run_py):
    assert run_py.is_file(), f"missing recipe entry point: {run_py}"
    with patch("espnet3.systems.launch.launch") as mock_launch:
        _exec_run_py(run_py, "espnet3_test_run_py_as_worker_import")
    mock_launch.assert_not_called()


def test_run_py_calls_launch_when_executed_as_a_script():
    """Running the file as a script (module name "__main__") calls launch()."""
    run_py = _RECIPE_RUN_PY_FILES[0]
    with patch("espnet3.systems.launch.launch") as mock_launch:
        _exec_run_py(run_py, "__main__")
    mock_launch.assert_called_once()
