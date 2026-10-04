"""Tests for the ESPnet3 SVS system stage hooks."""

import pytest
from omegaconf import OmegaConf

import espnet3.systems.base.training as training_module
import espnet3.systems.esp2_svs.system as sysmod
from espnet3.systems.esp2_svs.gan_lightning_module import GANLightningModule
from espnet3.systems.esp2_svs.system import SVSSystem


def _training_config(tmp_path):
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "stats_dir": str(tmp_path / "exp" / "stats"),
            "model": {
                "normalize": "global_mvn",
                "normalize_conf": {"stats_file": "missing.npz"},
            },
        }
    )


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


def test_collect_stats_drops_normalize(tmp_path, monkeypatch):
    seen = {}

    class FakeTrainer:
        def collect_stats(self):
            seen["called"] = True

    def fake_build_trainer(config, module_cls=None):
        seen["model"] = OmegaConf.to_container(config.model)
        return FakeTrainer()

    monkeypatch.setattr(training_module, "_build_trainer", fake_build_trainer)
    SVSSystem(training_config=_training_config(tmp_path)).collect_stats()
    assert seen["called"]
    assert seen["model"] == {}
