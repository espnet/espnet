"""GAN-TTS trainer helpers for ESPnet3 TTS systems."""

from __future__ import annotations

import copy

from omegaconf import DictConfig

from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.systems.tts.models.gan_model import GANTTSLightningModule


class GANTTSLightningTrainer(ESPnet3LightningTrainer):
    """ESPnet3 trainer wrapper that strips TTS GAN-only trainer config."""

    def __init__(
        self,
        model=None,
        exp_dir: str | None = None,
        config=None,
        best_model_criterion=None,
    ):
        trainer_config = copy.deepcopy(config)
        if isinstance(trainer_config, DictConfig) and hasattr(trainer_config, "gan"):
            delattr(trainer_config, "gan")
        elif isinstance(trainer_config, dict):
            trainer_config.pop("gan", None)

        super().__init__(
            model=model,
            exp_dir=exp_dir,
            config=trainer_config,
            best_model_criterion=best_model_criterion,
        )


def build_gan_trainer(training_config, model) -> GANTTSLightningTrainer:
    """Build the GAN-specific Lightning trainer for TTS."""
    lit_model = GANTTSLightningModule(model, training_config)
    return GANTTSLightningTrainer(
        model=lit_model,
        exp_dir=training_config.exp_dir,
        config=training_config.trainer,
        best_model_criterion=training_config.best_model_criterion,
    )
