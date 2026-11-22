import lightning
import pytest
import torch
from omegaconf import OmegaConf

import espnet3.components.trainers.trainer as trainer_mod
import espnet3.systems.base.training as train_mod
from espnet3.components.data import data_organizer as data_organizer_module
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.systems.base.training import _build_trainer, _instantiate_model
from espnet3.utils.config_utils import load_config_with_defaults

_UNSET = object()


class DummyTrainer:
    def __init__(self):
        self.collect_stats_called = False
        self.fit_called = False
        self.fit_kwargs = None

    def collect_stats(self):
        self.collect_stats_called = True

    def fit(self, **kwargs):
        self.fit_called = True
        self.fit_kwargs = kwargs


def test_ensure_directories_creates_exp_and_stats(tmp_path):
    cfg = OmegaConf.create(
        {"exp_dir": str(tmp_path / "exp"), "stats_dir": str(tmp_path / "stats")}
    )

    train_mod._ensure_directories(cfg)

    assert (tmp_path / "exp").is_dir()
    assert (tmp_path / "stats").is_dir()


def test_collect_stats_runs_pipeline(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "stats_dir": str(tmp_path / "stats"),
            "seed": 777,
            "parallel": {"backend": "dummy"},
            "model": {"normalize": True, "normalize_conf": {"foo": "bar"}},
        }
    )
    trainer = DummyTrainer()
    calls = {}

    def fake_set_parallel(arg):
        calls.setdefault("parallel", arg)

    def fake_seed_everything(seed, workers=True):
        calls.setdefault("seed", seed)

    def fake_set_precision(precision):
        calls.setdefault("precision", precision)

    monkeypatch.setattr(train_mod, "_build_trainer", lambda _cfg: trainer)
    monkeypatch.setattr(train_mod, "set_parallel", fake_set_parallel)
    monkeypatch.setattr(train_mod.L, "seed_everything", fake_seed_everything)
    monkeypatch.setattr(
        train_mod.torch, "set_float32_matmul_precision", fake_set_precision
    )

    train_mod.collect_stats(cfg)

    assert trainer.collect_stats_called
    assert calls["parallel"] == {"backend": "dummy"}
    assert calls["seed"] == 777
    assert calls["precision"] == "high"
    assert "normalize" not in cfg.model
    assert "normalize_conf" not in cfg.model


def test_train_saves_config_and_calls_fit(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "seed": 123,
            "parallel": {"backend": "dummy"},
            "task": "asr",
            "model": {"_target_": "dummy.Target"},
            "fit": {"max_epochs": 1},
        }
    )
    trainer = DummyTrainer()
    calls = {}

    def fake_set_parallel(arg):
        calls.setdefault("parallel", arg)

    def fake_seed_everything(seed, workers=True):
        calls.setdefault("seed", seed)

    def fake_set_precision(precision):
        calls.setdefault("precision", precision)

    def fake_save_config(task, cfg_arg, exp_dir):
        calls.setdefault("save", (task, exp_dir))

    monkeypatch.setattr(train_mod, "_build_trainer", lambda _cfg: trainer)
    monkeypatch.setattr(train_mod, "set_parallel", fake_set_parallel)
    monkeypatch.setattr(train_mod.L, "seed_everything", fake_seed_everything)
    monkeypatch.setattr(
        train_mod.torch, "set_float32_matmul_precision", fake_set_precision
    )
    monkeypatch.setattr(train_mod, "save_espnet_config", fake_save_config)

    train_mod.train(cfg)

    assert trainer.fit_called
    assert trainer.fit_kwargs == {"max_epochs": 1}
    assert calls["parallel"] == {"backend": "dummy"}
    assert calls["seed"] == 123
    assert calls["precision"] == "high"
    assert calls["save"] == ("asr", str(tmp_path / "exp"))


def test_instantiate_model_without_task_calls_instantiate(monkeypatch):
    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "dummy.Model",
                "freeze_param": ["frontend"],
            }
        }
    )
    calls = {}

    def fake_instantiate(config):
        calls["model"] = config
        return "dummy_model"

    monkeypatch.setattr(train_mod, "instantiate", fake_instantiate)
    result = _instantiate_model(cfg)

    assert result == "dummy_model"
    assert calls["model"]._target_ == "dummy.Model"
    assert "freeze_param" not in calls["model"]


def test_instantiate_model_with_task_calls_get_espnet_model(monkeypatch):
    cfg = OmegaConf.create(
        {
            "task": "asr",
            "model": {
                "_target_": "dummy.Model",
                "freeze_param": ["frontend"],
            },
        }
    )
    calls = {}

    def fake_get_espnet_model(task, model_config):
        calls["task"] = task
        calls["model_config"] = model_config
        return "espnet_model"

    monkeypatch.setattr(train_mod, "get_espnet_model", fake_get_espnet_model)
    result = _instantiate_model(cfg)

    assert result == "espnet_model"
    assert calls["task"] == "asr"
    assert calls["model_config"]["_target_"] == "dummy.Model"
    assert "freeze_param" not in calls["model_config"]


def test_build_trainer_assembles_components(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "trainer": {"max_epochs": 1},
            "best_model_criterion": {"monitor": "valid/loss", "mode": "min"},
            "model": {"_target_": "dummy.Model"},
        }
    )
    calls = {}

    def fake_instantiate_model(config):
        return "dummy_model"

    def fake_lit_module(model, config):
        calls["model"] = model
        return "lit_model"

    def fake_trainer(*, model, exp_dir, config, best_model_criterion):
        calls["trainer_args"] = {
            "model": model,
            "exp_dir": exp_dir,
        }
        return DummyTrainer()

    monkeypatch.setattr(train_mod, "_instantiate_model", fake_instantiate_model)
    monkeypatch.setattr(train_mod, "ESPnetLightningModule", fake_lit_module)
    monkeypatch.setattr(train_mod, "ESPnet3LightningTrainer", fake_trainer)

    trainer = _build_trainer(cfg)

    assert isinstance(trainer, DummyTrainer)
    assert calls["model"] == "dummy_model"
    assert calls["trainer_args"]["model"] == "lit_model"
    assert calls["trainer_args"]["exp_dir"] == str(tmp_path / "exp")


class _InitDummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return {"x": torch.zeros(3)}

    def __len__(self):
        return 2


class _PretrainedFrontend(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.proj = torch.nn.Linear(in_features, in_features)
        self.pretrained = {k: v.clone() for k, v in self.state_dict().items()}

    def reload_pretrained_parameters(self):
        self.load_state_dict(self.pretrained)


class _PretrainedFrontendModel(torch.nn.Module):
    """Model whose frontend restores its own weights after ``initialize()``."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.frontend = _PretrainedFrontend(in_features)
        self.head = torch.nn.Linear(in_features, out_features)


_INIT_TRAINING_YAML = """
exp_dir: {exp_dir}
model:
  _target_: {model_target}
  in_features: 3
  out_features: 1
optim:
  _target_: torch.optim.Adam
  lr: 0.001
scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: 1
dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  train:
    - name: train_dummy
      data_src: dummy
  valid:
    - name: valid_dummy
      data_src: dummy
dataloader:
  train:
    batch_size: 1
  valid:
    batch_size: 1
best_model_criterion:
  - - valid/loss
    - 1
    - min
trainer:
  accelerator: cpu
  devices: 1
  log_every_n_steps: 1
  max_epochs: 1
"""


@pytest.fixture
def load_init_training_config(tmp_path, monkeypatch):
    """Load a training YAML through the recipe config loader."""
    monkeypatch.setattr(
        data_organizer_module,
        "instantiate_dataset_reference",
        lambda config, recipe_dir=None: _InitDummyDataset(),
    )

    def load(top_init=_UNSET, trainer_init=_UNSET, model_target="torch.nn.Linear"):
        text = _INIT_TRAINING_YAML.format(
            exp_dir=tmp_path / "exp", model_target=model_target
        )
        if trainer_init is not _UNSET:
            text += f"  init: {trainer_init or 'null'}\n"
        if top_init is not _UNSET:
            text += f"init: {top_init or 'null'}\n"
        path = tmp_path / "training.yaml"
        path.write_text(text)
        return load_config_with_defaults(str(path))

    return load


@pytest.fixture
def record_initialize(monkeypatch):
    calls = []
    real_initialize = trainer_mod.initialize

    def fake_initialize(model, init):
        assert not isinstance(model, ESPnetLightningModule)
        calls.append(init)
        real_initialize(model, init)

    monkeypatch.setattr(trainer_mod, "initialize", fake_initialize)
    return calls


@pytest.mark.parametrize(
    "top_init, trainer_init",
    [
        (_UNSET, "xavier_uniform"),
        ("xavier_uniform", _UNSET),
        ("xavier_uniform", "xavier_uniform"),
        (None, "xavier_uniform"),
    ],
)
def test_build_trainer_applies_init_without_leaking_to_lightning(
    top_init, trainer_init, load_init_training_config, record_initialize
):
    config = load_init_training_config(top_init=top_init, trainer_init=trainer_init)
    original = OmegaConf.to_container(config, resolve=False)

    trainer = _build_trainer(config)

    assert record_initialize == ["xavier_uniform"]
    assert isinstance(trainer.trainer, lightning.Trainer)
    assert OmegaConf.to_container(config, resolve=False) == original


def test_build_trainer_init_keeps_pretrained_frontend_weights(
    load_init_training_config, record_initialize
):
    config = load_init_training_config(
        top_init="xavier_uniform",
        model_target=f"{__name__}._PretrainedFrontendModel",
    )

    trainer = _build_trainer(config)

    frontend = trainer.model.model.frontend
    assert record_initialize == ["xavier_uniform"]
    assert torch.equal(frontend.proj.weight, frontend.pretrained["proj.weight"])
    assert torch.equal(frontend.proj.bias, frontend.pretrained["proj.bias"])


@pytest.mark.parametrize(
    "top_init, trainer_init",
    [(_UNSET, _UNSET), (None, None), (_UNSET, None), (None, _UNSET)],
)
def test_build_trainer_skips_missing_or_null_init(
    top_init, trainer_init, load_init_training_config, record_initialize
):
    config = load_init_training_config(top_init=top_init, trainer_init=trainer_init)

    trainer = _build_trainer(config)

    assert record_initialize == []
    assert isinstance(trainer.trainer, lightning.Trainer)


def test_build_trainer_rejects_conflicting_init(
    load_init_training_config, record_initialize
):
    config = load_init_training_config(
        top_init="xavier_uniform", trainer_init="kaiming_normal"
    )

    with pytest.raises(ValueError, match="init='xavier_uniform'.*kaiming_normal"):
        _build_trainer(config)
    assert record_initialize == []
