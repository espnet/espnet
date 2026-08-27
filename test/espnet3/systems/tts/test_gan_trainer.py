"""Tests for the GAN-TTS trainer wrapper used by the VITS recipe."""

import pytest
from omegaconf import OmegaConf

import espnet3.systems.tts.gan_trainer as gan_trainer_module
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.systems.tts.gan_trainer import GANTTSLightningTrainer, build_gan_trainer

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_strips_gan_section_from_dictconfig     | trainer.gan never reaches    |
# |                                             | the Lightning trainer.       |
# | test_strips_gan_section_from_plain_dict     | Same for a plain dict.       |
# | test_does_not_mutate_caller_config          | The caller's config keeps    |
# |                                             | its gan section.             |
# | test_forwards_config_without_gan_section    | A gan-free config is passed  |
# |                                             | through untouched.           |
# | test_build_gan_trainer_wires_config         | build_gan_trainer wraps the  |
# |                          | model and forwards exp_dir/criterion/trainer.    |


@pytest.fixture
def recorded_init(monkeypatch):
    """Record what GANTTSLightningTrainer hands to the base Lightning trainer."""
    calls = []

    def fake_init(
        self, model=None, exp_dir=None, config=None, best_model_criterion=None
    ):
        calls.append(
            {
                "model": model,
                "exp_dir": exp_dir,
                "config": config,
                "best_model_criterion": best_model_criterion,
            }
        )

    monkeypatch.setattr(ESPnet3LightningTrainer, "__init__", fake_init)
    return calls


def test_strips_gan_section_from_dictconfig(recorded_init):
    config = OmegaConf.create({"accelerator": "cpu", "gan": {"generator_first": True}})

    GANTTSLightningTrainer(config=config)

    forwarded = recorded_init[0]["config"]
    assert "gan" not in forwarded
    assert forwarded.accelerator == "cpu"


def test_strips_gan_section_from_plain_dict(recorded_init):
    config = {"accelerator": "cpu", "gan": {"generator_first": True}}

    GANTTSLightningTrainer(config=config)

    assert recorded_init[0]["config"] == {"accelerator": "cpu"}


def test_does_not_mutate_caller_config(recorded_init):
    """The strip happens on a deep copy; the recipe config stays intact."""
    dict_config = OmegaConf.create({"accelerator": "cpu", "gan": {"ratio": 2}})
    plain_config = {"accelerator": "cpu", "gan": {"ratio": 2}}

    GANTTSLightningTrainer(config=dict_config)
    GANTTSLightningTrainer(config=plain_config)

    assert "gan" in dict_config
    assert "gan" in plain_config


def test_forwards_config_without_gan_section(recorded_init):
    config = OmegaConf.create({"accelerator": "cpu"})

    GANTTSLightningTrainer(config=config)

    assert recorded_init[0]["config"] == config


def test_build_gan_trainer_wires_config(recorded_init, monkeypatch):
    wrapped = []

    class _FakeLightningModule:
        def __init__(self, model, config):
            wrapped.append((model, config))

    monkeypatch.setattr(
        gan_trainer_module, "GANTTSLightningModule", _FakeLightningModule
    )
    model = object()
    training_config = OmegaConf.create(
        {
            "exp_dir": "exp/vits",
            "best_model_criterion": [["valid/loss", 3, "min"]],
            "trainer": {"accelerator": "cpu", "gan": {"generator_first": True}},
        }
    )

    trainer = build_gan_trainer(training_config, model)

    assert isinstance(trainer, GANTTSLightningTrainer)
    assert wrapped == [(model, training_config)]
    call = recorded_init[0]
    assert isinstance(call["model"], _FakeLightningModule)
    assert call["exp_dir"] == "exp/vits"
    assert call["best_model_criterion"] == training_config.best_model_criterion
    assert "gan" not in call["config"]
