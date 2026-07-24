"""Unit tests for espnet3.systems.codec.system."""

import pytest
import torch.nn as nn
from omegaconf import OmegaConf

import espnet3.systems.codec.system as sysmod
from espnet3.systems.codec.system import CodecSystem, _instantiate_model

from ._gan_dummies import DummyGANModel


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


def patch_runtime(monkeypatch, trainer):
    calls = {}
    monkeypatch.setattr(CodecSystem, "_build_trainer", lambda self: trainer)
    monkeypatch.setattr(
        sysmod, "set_parallel", lambda arg: calls.setdefault("parallel", arg)
    )
    monkeypatch.setattr(
        sysmod.L,
        "seed_everything",
        lambda seed, workers=True: calls.setdefault("seed", seed),
    )
    monkeypatch.setattr(
        sysmod.torch,
        "set_float32_matmul_precision",
        lambda p: calls.setdefault("precision", p),
    )
    return calls


# ---------------- _instantiate_model ----------------


def test_instantiate_model_with_task_uses_espnet_task(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        sysmod,
        "get_espnet_model",
        lambda task, conf: calls.setdefault("args", (task, conf)) and "espnet_model",
    )
    cfg = OmegaConf.create({"task": "codec", "model": {"foo": 1}})
    assert _instantiate_model(cfg) == "espnet_model"
    assert calls["args"] == ("codec", {"foo": 1})


def test_instantiate_model_without_task_uses_hydra(monkeypatch):
    monkeypatch.setattr(sysmod, "instantiate", lambda conf: "hydra_model")
    cfg = OmegaConf.create({"model": {"_target_": "dummy.Model"}})
    assert _instantiate_model(cfg) == "hydra_model"


# ---------------- directories ----------------


def test_ensure_directories_creates_exp_and_stats(tmp_path):
    cfg = OmegaConf.create(
        {"exp_dir": str(tmp_path / "exp"), "stats_dir": str(tmp_path / "stats")}
    )
    CodecSystem(training_config=cfg)._ensure_directories()
    assert (tmp_path / "exp").is_dir()
    assert (tmp_path / "stats").is_dir()


def test_ensure_directories_without_stats_dir(tmp_path):
    cfg = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    CodecSystem(training_config=cfg)._ensure_directories()
    assert (tmp_path / "exp").is_dir()


# ---------------- _build_trainer dispatch ----------------


def test_build_trainer_uses_gan_path_for_gan_model(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {"exp_dir": str(tmp_path / "exp"), "model": {"_target_": "d.M"}}
    )
    gan_model = DummyGANModel()
    monkeypatch.setattr(sysmod, "_instantiate_model", lambda c: gan_model)
    built = {}
    monkeypatch.setattr(
        sysmod,
        "build_gan_trainer",
        lambda config, model: built.setdefault("args", (config, model))
        and "gan_trainer",
    )
    system = CodecSystem(training_config=cfg)
    assert system._build_trainer() == "gan_trainer"
    assert built["args"][1] is gan_model


def test_build_trainer_uses_plain_path_for_non_gan_model(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "model": {"_target_": "d.M"},
            "trainer": {"max_epochs": 1},
            "best_model_criterion": None,
        }
    )
    monkeypatch.setattr(sysmod, "_instantiate_model", lambda c: nn.Linear(1, 1))
    monkeypatch.setattr(
        sysmod, "ESPnetLightningModule", lambda model, config: "lit_model"
    )
    recorded = {}

    def fake_trainer(model=None, exp_dir=None, config=None, best_model_criterion=None):
        recorded.update(model=model, exp_dir=exp_dir)
        return "plain_trainer"

    monkeypatch.setattr(sysmod, "ESPnet3LightningTrainer", fake_trainer)
    system = CodecSystem(training_config=cfg)
    assert system._build_trainer() == "plain_trainer"
    assert recorded["model"] == "lit_model"
    assert recorded["exp_dir"] == str(tmp_path / "exp")


# ---------------- stages ----------------


def test_collect_stats_runs_pipeline(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "stats_dir": str(tmp_path / "stats"),
            "seed": 777,
            "parallel": {"backend": "dummy"},
        }
    )
    trainer = DummyTrainer()
    calls = patch_runtime(monkeypatch, trainer)
    system = CodecSystem(training_config=cfg)

    system.collect_stats()

    assert trainer.collect_stats_called
    assert calls["parallel"] == cfg.parallel
    assert calls["seed"] == 777
    assert calls["precision"] == "high"
    assert (tmp_path / "exp").is_dir()


def test_train_saves_espnet_config_and_forwards_fit_kwargs(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "seed": 123,
            "task": "codec",
            "fit": {"ckpt_path": "last"},
        }
    )
    trainer = DummyTrainer()
    calls = patch_runtime(monkeypatch, trainer)
    monkeypatch.setattr(
        sysmod,
        "save_espnet_config",
        lambda task, c, exp_dir: calls.setdefault("save", (task, exp_dir)),
    )
    system = CodecSystem(training_config=cfg)

    system.train()

    assert trainer.fit_called
    assert trainer.fit_kwargs == {"ckpt_path": "last"}
    assert calls["save"] == ("codec", str(tmp_path / "exp"))
    assert calls["seed"] == 123


def test_train_without_task_and_fit(tmp_path, monkeypatch):
    cfg = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    trainer = DummyTrainer()
    calls = patch_runtime(monkeypatch, trainer)
    monkeypatch.setattr(
        sysmod,
        "save_espnet_config",
        lambda task, c, exp_dir: calls.setdefault("save", True),
    )
    system = CodecSystem(training_config=cfg)

    system.train()

    assert trainer.fit_called
    assert trainer.fit_kwargs == {}
    assert "save" not in calls
    # no parallel / seed configured -> hooks not invoked
    assert "parallel" not in calls
    assert "seed" not in calls


@pytest.mark.parametrize("stage", ["train", "collect_stats"])
def test_stages_reject_positional_arguments(tmp_path, stage):
    cfg = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    system = CodecSystem(training_config=cfg)
    with pytest.raises(TypeError, match="does not accept arguments"):
        getattr(system, stage)("unexpected")
