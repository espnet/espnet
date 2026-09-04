"""Unit tests for espnet3.systems.codec.gan_trainer."""

from test.espnet3.systems.codec._gan_dummies import (
    DummyGANModel,
    make_gan_training_config,
)

from omegaconf import OmegaConf

import espnet3.systems.codec.gan_trainer as gt
from espnet3.systems.codec.models.gan_lightning_module import GANLightningModule


def _capture_parent_init(monkeypatch):
    recorded = {}

    def fake_init(
        self, model=None, exp_dir=None, config=None, best_model_criterion=None
    ):
        recorded.update(
            model=model,
            exp_dir=exp_dir,
            config=config,
            best_model_criterion=best_model_criterion,
        )

    monkeypatch.setattr(gt.ESPnet3LightningTrainer, "__init__", fake_init)
    return recorded


def test_strips_gan_key_from_dictconfig(monkeypatch):
    recorded = _capture_parent_init(monkeypatch)
    config = OmegaConf.create({"max_epochs": 3, "gan": {"generator_first": True}})
    gt.GANLightningTrainer(
        model="m", exp_dir="exp", config=config, best_model_criterion="crit"
    )
    assert "gan" not in recorded["config"]
    assert recorded["config"].max_epochs == 3
    assert recorded["model"] == "m"
    assert recorded["exp_dir"] == "exp"
    assert recorded["best_model_criterion"] == "crit"
    # original config must not be mutated
    assert "gan" in config


def test_strips_gan_key_from_plain_dict(monkeypatch):
    recorded = _capture_parent_init(monkeypatch)
    config = {"max_epochs": 3, "gan": {"skip_discriminator_prob": 0.5}}
    gt.GANLightningTrainer(model="m", exp_dir="exp", config=config)
    assert "gan" not in recorded["config"]
    assert recorded["config"]["max_epochs"] == 3
    assert "gan" in config


def test_config_without_gan_key_passes_through(monkeypatch):
    recorded = _capture_parent_init(monkeypatch)
    config = OmegaConf.create({"max_epochs": 3})
    gt.GANLightningTrainer(model="m", exp_dir="exp", config=config)
    assert recorded["config"].max_epochs == 3


def test_build_gan_trainer_wires_module_and_trainer(monkeypatch):
    created = {}

    class FakeModule:
        def __init__(self, model, config):
            created["module"] = (model, config)

    class FakeTrainer:
        def __init__(
            self, model=None, exp_dir=None, config=None, best_model_criterion=None
        ):
            created["trainer"] = {
                "model": model,
                "exp_dir": exp_dir,
                "config": config,
                "best_model_criterion": best_model_criterion,
            }

    monkeypatch.setattr(gt, "GANLightningModule", FakeModule)
    monkeypatch.setattr(gt, "GANLightningTrainer", FakeTrainer)

    training_config = OmegaConf.create(
        {
            "exp_dir": "exp",
            "trainer": {"max_epochs": 1},
            "best_model_criterion": [["valid/loss", 3, "min"]],
        }
    )
    out = gt.build_gan_trainer(training_config, model="raw_model")

    assert created["module"][0] == "raw_model"
    assert created["module"][1] is training_config
    assert isinstance(created["trainer"]["model"], FakeModule)
    assert created["trainer"]["exp_dir"] == "exp"
    assert created["trainer"]["config"] is training_config.trainer
    assert isinstance(out, FakeTrainer)


# ---------------- real trainer integration ----------------


def test_gan_fit_runs_one_batch_with_real_trainer(tmp_path):
    from espnet3.systems.codec.gan_trainer import GANLightningTrainer

    model = GANLightningModule(
        DummyGANModel(), make_gan_training_config(tmp_path / "exp")
    )
    trainer_config = OmegaConf.create(
        {
            "accelerator": "cpu",
            "devices": 1,
            "num_nodes": 1,
            "max_epochs": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 0,
            "num_sanity_val_steps": 0,
            "log_every_n_steps": 1,
            "logger": {
                "_target_": "lightning.pytorch.loggers.TensorBoardLogger",
                "save_dir": str(tmp_path / "tb"),
                "name": "gan_test",
            },
            "gradient_clip_val": 0.0,
            # must be stripped by GANLightningTrainer before reaching Lightning
            "gan": {"generator_first": True},
        }
    )
    trainer = GANLightningTrainer(
        model=model,
        exp_dir=str(tmp_path / "exp"),
        config=trainer_config,
    )
    trainer.fit()
    assert model._optimizer_states["generator"].update_step == 1
    assert model._optimizer_states["discriminator"].update_step == 1
