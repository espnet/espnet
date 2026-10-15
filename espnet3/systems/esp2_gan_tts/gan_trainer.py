"""GAN-TTS trainer helpers for the ESPnet3 esp2_gan_tts system."""

from __future__ import annotations

import copy

from omegaconf import DictConfig

from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.systems.esp2_gan_tts.models.gan_model import GANTTSLightningModule


class GANTTSLightningTrainer(ESPnet3LightningTrainer):
    """ESPnet3 trainer wrapper for GAN-TTS models.

    ``ESPnet3LightningTrainer`` forwards every key of the ``trainer:`` config
    block to ``lightning.Trainer(...)``. GAN-TTS recipes carry an extra
    ``trainer.gan`` sub-block (``generator_first``,
    ``skip_discriminator_prob``, ``no_forward_run``) that is consumed by
    :class:`~espnet3.systems.esp2_gan_tts.models.gan_model.GANTTSLightningModule`,
    not by Lightning. This subclass strips that block before delegating, so
    Lightning never sees an unknown ``gan`` keyword. Everything else -
    callbacks, checkpoint averaging, ``fit``/``collect_stats`` - is inherited
    unchanged.

    Examples:
        Built by :func:`build_gan_trainer` from a recipe config such as:
        ```yaml
        trainer:
          accelerator: auto
          max_epochs: 1000
          gan:
            generator_first: false
            skip_discriminator_prob: 0.0
        ```
    """

    def __init__(
        self,
        model=None,
        exp_dir: str | None = None,
        config=None,
        best_model_criterion=None,
    ):
        """Initialize GANTTSLightningTrainer, stripping GAN-specific config keys.

        Removes the ``gan`` sub-config from *config* before delegating to the
        parent ``ESPnet3LightningTrainer``, so GAN-only keys (e.g. the
        generator/discriminator turn order) do not interfere with the base
        Lightning trainer.

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

        Notes:
            The ``gan`` key is stripped on a deep copy of *config*, so the
            caller's object is never mutated. The parent constructor reads
            ``model.config.dataloader`` and ``config.log_every_n_steps``, so
            *model* must be a real Lightning module built from a full
            training config; use :func:`build_gan_trainer` rather than
            calling this constructor directly.

        Examples:
            ```python
            from omegaconf import OmegaConf

            trainer = build_gan_trainer(training_config, model)
            "gan" in training_config.trainer  # -> True  (caller untouched)
            "gan" in trainer.config  # -> False (stripped copy)
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


def build_gan_trainer(training_config, model) -> GANTTSLightningTrainer:
    """Build the GAN-specific Lightning trainer for an espnet2 GAN-TTS model.

    Wraps *model* in a :class:`GANTTSLightningModule` (manual two-optimizer
    optimization) and hands it to :class:`GANTTSLightningTrainer`. This is
    the function ``GANTTSSystem.collect_stats`` / ``train`` dispatch to when
    the instantiated model is an ``AbsGANESPnetModel``.

    Args:
        training_config: The full training config (``DictConfig``). Keys
            read here: ``exp_dir``, ``trainer`` (including the optional
            ``trainer.gan`` block) and ``best_model_criterion``. The module
            additionally reads ``optimizers`` / ``schedulers`` (named
            ``generator`` / ``discriminator`` entries) and ``dataloader``.
        model: An instantiated espnet2 GAN-TTS model
            (``espnet2.gan_tts.espnet_model.ESPnetGANTTSModel``, e.g. VITS).

    Returns:
        GANTTSLightningTrainer: Ready for ``.collect_stats()`` or ``.fit()``.

    Examples:
        The corresponding ``training.yaml`` fragment:
        ```yaml
        task: espnet3.systems.esp2_gan_tts.task.GANTTSTask
        optimizers:
          generator:
            optimizer: {_target_: torch.optim.AdamW, lr: 2.0e-4}
            params: generator
          discriminator:
            optimizer: {_target_: torch.optim.AdamW, lr: 2.0e-4}
            params: discriminator
        trainer:
          accelerator: auto
          gan:
            generator_first: false
        ```
        and the call ``GANTTSSystem.train`` makes:
        ```python
        trainer = build_gan_trainer(training_config, model)
        trainer.fit()
        ```
    """
    lit_model = GANTTSLightningModule(model, training_config)
    return GANTTSLightningTrainer(
        model=lit_model,
        exp_dir=training_config.exp_dir,
        config=training_config.trainer,
        best_model_criterion=training_config.best_model_criterion,
    )
