from types import SimpleNamespace

from omegaconf import OmegaConf

import espnet3.systems.tts.gan_trainer as gan_train_mod


def test_gan_tts_lightning_trainer_strips_gan_config(monkeypatch):
    calls = {}

    def fake_super_init(self, model, exp_dir, config, best_model_criterion):
        calls["model"] = model
        calls["exp_dir"] = exp_dir
        calls["config"] = config
        calls["best"] = best_model_criterion

    monkeypatch.setattr(
        gan_train_mod.ESPnet3LightningTrainer,
        "__init__",
        fake_super_init,
    )

    config = OmegaConf.create(
        {
            "accelerator": "cpu",
            "devices": 1,
            "gan": {"generator_first": True},
        }
    )
    trainer = gan_train_mod.GANTTSLightningTrainer(
        model="model",
        exp_dir="exp",
        config=config,
        best_model_criterion=[("valid/loss", 1, "min")],
    )

    assert trainer is not None
    assert not hasattr(calls["config"], "gan")
    assert hasattr(config, "gan")


def test_build_gan_trainer_uses_gan_module_and_wrapper(monkeypatch):
    calls = {}

    class DummyLit:
        def __init__(self, model, config):
            calls["lit"] = (model, config)

    class DummyTrainer:
        def __init__(self, model, exp_dir, config, best_model_criterion):
            calls["trainer"] = (model, exp_dir, config, best_model_criterion)

    monkeypatch.setattr(gan_train_mod, "GANTTSLightningModule", DummyLit)
    monkeypatch.setattr(gan_train_mod, "GANTTSLightningTrainer", DummyTrainer)

    training_config = SimpleNamespace(
        exp_dir="exp",
        trainer=OmegaConf.create({"accelerator": "cpu"}),
        best_model_criterion=[("valid/generator/loss", 1, "min")],
    )

    result = gan_train_mod.build_gan_trainer(training_config, model="gan-model")

    assert isinstance(result, DummyTrainer)
    assert calls["lit"] == ("gan-model", training_config)
    assert calls["trainer"][1:] == (
        "exp",
        training_config.trainer,
        training_config.best_model_criterion,
    )
