"""Shared GAN trainer helpers for ESPnet3.

Pairs with ``espnet3.systems.codec.models.gan_lightning_module.GANLightningModule``.
Kept task-agnostic so both GAN-TTS and GAN-based neural codec systems can
reuse it without duplication.
"""

from __future__ import annotations

import copy

from omegaconf import DictConfig

from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.systems.codec.models.gan_lightning_module import GANLightningModule


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
        interfere with the base Lightning trainer, which splats the rest of
        the block into ``lightning.Trainer(**config)``.

        Args:
            model: The Lightning module to train, normally a
                :class:`GANLightningModule`.
            exp_dir: Experiment directory for checkpoints and logs.
            config: The recipe's ``trainer`` block. May contain a ``gan``
                sub-block, which is removed here rather than mutated in
                place -- *config* itself is left untouched.
            best_model_criterion: Checkpoint selection rules, each of the
                form ``[monitor_key, num_to_keep, "min" | "max"]``.

        Examples:
            Usually reached through :func:`build_gan_trainer` rather than
            constructed directly:
            ```python
            trainer = GANLightningTrainer(
                model=GANLightningModule(model, config),
                exp_dir=config.exp_dir,
                config=config.trainer,          # may include `gan:`
                best_model_criterion=config.best_model_criterion,
            )
            ```
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
    """Build the shared GAN Lightning trainer for any AbsGANESPnetModel.

    Wraps *model* in a :class:`GANLightningModule` (which drives the manual
    generator/discriminator optimization loop) and hands it to a
    :class:`GANLightningTrainer` configured from the recipe's training config.

    Args:
        training_config: Resolved training config. The following fields are
            read: ``trainer`` (Lightning trainer kwargs plus the GAN-only
            ``gan`` sub-block), ``exp_dir`` (checkpoint/log destination),
            ``best_model_criterion`` (checkpoint selection rules), and the
            ``optimizers``/``schedulers`` blocks consumed by
            ``GANLightningModule``.
        model: The GAN model to train. Expected to be an
            ``espnet2.train.abs_gan_espnet_model.AbsGANESPnetModel``, i.e. a
            model exposing ``generator`` and ``discriminator`` submodules and
            a ``forward(..., forward_generator=...)`` signature.

    Returns:
        GANLightningTrainer: A trainer ready for ``.fit()`` or
        ``.collect_stats()``.

    Examples:
        ```python
        from espnet3.systems.codec.gan_trainer import build_gan_trainer
        from espnet3.utils.task_utils import get_espnet_model

        model = get_espnet_model(config.task, config.model)
        trainer = build_gan_trainer(config, model)
        trainer.fit()
        ```

        ``CodecSystem`` picks this path automatically for GAN models:
        ```python
        if isinstance(model, AbsGANESPnetModel):
            return build_gan_trainer(config, model)
        ```
    """
    lit_model = GANLightningModule(model, training_config)
    return GANLightningTrainer(
        model=lit_model,
        exp_dir=training_config.exp_dir,
        config=training_config.trainer,
        best_model_criterion=training_config.best_model_criterion,
    )
