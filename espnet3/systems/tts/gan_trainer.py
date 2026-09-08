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
        """Initialize GANTTSLightningTrainer, stripping GAN-specific config keys.

        Removes the ``gan`` sub-config from *config* before delegating to the
        parent ``ESPnet3LightningTrainer``, so GAN-only keys (e.g. discriminator
        update intervals) do not interfere with the base Lightning trainer.

        Args:
            model: The Lightning module to train.  Typically a
                ``GANTTSLightningModule`` instance.
            exp_dir: Path to the experiment output directory where checkpoints
                and logs are written.  ``None`` disables checkpoint saving.
            config: Trainer configuration (OmegaConf ``DictConfig`` or plain
                ``dict``).  The ``gan`` key, if present, is stripped before use.
            best_model_criterion: Sequence of ``(metric, weight, mode)`` triples
                used to select the best checkpoint.  Pass ``None`` to disable.

        Returns:
            None

        Raises:
            TypeError: If *config* is neither a ``DictConfig`` nor a ``dict``,
                and the parent class cannot accept it.

        Notes:
            The ``gan`` key is stripped on a deep copy of *config*, so the
            caller's object is never mutated.

        Examples:
            >>> import torch.nn as nn
            >>> from omegaconf import OmegaConf
            >>> cfg = OmegaConf.create({"accelerator": "cpu", "gan": {"ratio": 2}})
            >>> trainer = GANTTSLightningTrainer(config=cfg)
            >>> "gan" not in trainer  # gan key was stripped
            True
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


def build_gan_trainer(training_config, model) -> GANTTSLightningTrainer:
    """Build the GAN-specific Lightning trainer for TTS."""
    lit_model = GANTTSLightningModule(model, training_config)
    return GANTTSLightningTrainer(
        model=lit_model,
        exp_dir=training_config.exp_dir,
        config=training_config.trainer,
        best_model_criterion=training_config.best_model_criterion,
    )
