"""Tests for the ESPnet3 SVS system stage hooks."""

import pytest
from omegaconf import OmegaConf

import espnet3.systems.base.training as training_module
import espnet3.systems.svs.system as sysmod
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.systems.svs.gan_lightning_module import GANLightningModule
from espnet3.systems.svs.system import SVSSystem


def _training_config(tmp_path, **extra):
    config = {
        "exp_dir": str(tmp_path / "exp"),
        "stats_dir": str(tmp_path / "exp" / "stats"),
        "model": {
            "normalize": "global_mvn",
            "normalize_conf": {"stats_file": "missing.npz"},
        },
        "trainer": {},
        "best_model_criterion": [],
    }
    config.update(extra)
    return OmegaConf.create(config)


def test_train_uses_gan_module(tmp_path, monkeypatch):
    calls = {}

    def fake_train(config, module_cls=None):
        calls["config"] = config
        calls["module_cls"] = module_cls

    monkeypatch.setattr(sysmod, "train", fake_train)
    system = SVSSystem(training_config=_training_config(tmp_path))
    system.train()
    assert calls["module_cls"] is GANLightningModule
    assert calls["config"] is system.training_config


def test_train_rejects_stage_args(tmp_path):
    system = SVSSystem(training_config=_training_config(tmp_path))
    with pytest.raises(TypeError, match="does not accept arguments"):
        system.train("unexpected")


class _FakeTrainer:
    def __init__(self):
        self.called = False

    def collect_stats(self):
        self.called = True


def test_collect_stats_drops_normalize(tmp_path, monkeypatch):
    seen = {}
    trainer = _FakeTrainer()

    def fake_build_trainer(config, module_cls=None):
        seen["model"] = OmegaConf.to_container(config.model)
        return trainer

    monkeypatch.setattr(training_module, "_build_trainer", fake_build_trainer)
    system = SVSSystem(training_config=_training_config(tmp_path))
    system.collect_stats()
    assert trainer.called
    assert "normalize" not in seen["model"]
    assert "normalize_conf" not in seen["model"]


def test_collect_stats_writes_feats_when_asked(tmp_path, monkeypatch):
    seen = {}

    def fake_collect_stats(**kwargs):
        seen["write_collected_feats"] = kwargs["write_collected_feats"]

    monkeypatch.setattr(
        "espnet3.components.modeling.lightning_module.collect_stats",
        fake_collect_stats,
    )
    monkeypatch.setattr(
        ESPnetLightningModule, "__init__", lambda self, model, config: None
    )
    for flag in (True, False):
        config = _training_config(
            tmp_path,
            dataset={},
            dataloader={},
            write_collected_feats=flag,
        )
        module = ESPnetLightningModule(None, config)
        module.config = config
        module.collect_stats()
        assert seen["write_collected_feats"] is flag
