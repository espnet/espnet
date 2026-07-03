"""Shared GAN trainer helpers for ESPnet3.

Pairs with ``espnet3.components.modeling.gan_lightning_module.GANLightningModule``.
Kept task-agnostic so both GAN-TTS and GAN-based neural codec systems can
reuse it without duplication.
"""

from __future__ import annotations

import copy

from omegaconf import DictConfig

from espnet3.components.modeling.gan_lightning_module import GANLightningModule
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer


class GANLightningTrainer(ESPnet3LightningTrainer):
    """ESPnet3 trainer wrapper that strips GAN-only trainer config."""

    def __init__(
        self,
        model=None,
        exp_dir: str | None = None,
        config=None,
        best_model_criterion=None,
    ):
        """Initialize GANLightningTrainer, stripping GAN-specific config keys.

        Removes the ``gan`` sub-config from *config* before delegating to the
        parent ``ESPnet3LightningTrainer``, so GAN-only keys (e.g. generator/
        discriminator turn order, discriminator skip probability) do not
        interfere with the base Lightning trainer.
        """
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


def build_gan_trainer(training_config, model) -> GANLightningTrainer:
    """Build the shared GAN Lightning trainer for any AbsGANESPnetModel."""
    lit_model = GANLightningModule(model, training_config)
    return GANLightningTrainer(
        model=lit_model,
        exp_dir=training_config.exp_dir,
        config=training_config.trainer,
        best_model_criterion=training_config.best_model_criterion,
    )
