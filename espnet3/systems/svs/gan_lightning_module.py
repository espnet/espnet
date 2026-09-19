"""Lightning module running ESPnet2 GAN models in generator/discriminator turns."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.modeling.optimization_spec import OptimizationStep


class GANLightningModule(ESPnetLightningModule):
    """Train an ``AbsGANESPnetModel`` with ESPnet3's named optimizers.

    ESPnet2 GAN models (VISinger, VITS, JETS, ...) do not return the
    ``(loss, stats, weight)`` tuple of the base module. Each call takes
    ``forward_generator`` and returns a dict with ``loss``, ``stats``,
    ``weight`` and ``optim_idx`` (0 for the generator, 1 for the
    discriminator). This module runs both turns for every batch, as
    ``espnet2.train.gan_trainer.GANTrainer`` does, and hands each loss to the
    named optimizer through the base module's multi-optimizer path, so
    ``config.optimizers`` must define ``generator`` and ``discriminator``.
    ``config.generator_first`` picks the turn order (default: discriminator
    first, the espnet2 default). Models that are not GAN models fall back to
    the base step unchanged.

    Example:
        ```yaml
        generator_first: true
        optimizers:
          generator:
            optimizer: {_target_: torch.optim.AdamW, lr: 2.0e-4}
            params: generator
          discriminator:
            optimizer: {_target_: torch.optim.AdamW, lr: 2.0e-4}
            params: discriminator
        ```
    """

    _OPTIMIZER_NAMES = {0: "generator", 1: "discriminator"}

    def _gan_turn(
        self, batch, forward_generator: bool
    ) -> Tuple[OptimizationStep, Dict[str, torch.Tensor], torch.Tensor | None]:
        """Run one generator or discriminator turn and wrap its loss."""
        output = self.model(**batch[1], forward_generator=forward_generator)
        if not isinstance(output, dict) or "optim_idx" not in output:
            raise TypeError(
                "GAN models must return a dict with loss/stats/weight/optim_idx, "
                f"but got {type(output)!r}."
            )
        optim_idx = int(output["optim_idx"])
        if optim_idx not in self._OPTIMIZER_NAMES:
            raise ValueError(
                f"optim_idx must be 0 (generator) or 1 (discriminator): {optim_idx}"
            )
        step = OptimizationStep(
            loss=output["loss"], name=self._OPTIMIZER_NAMES[optim_idx]
        )
        return step, output["stats"], output.get("weight")

    def _step(self, batch, batch_idx, mode):
        """Run the generator and discriminator turns for one batch."""
        if not isinstance(self.model, AbsGANESPnetModel):
            return super()._step(batch, batch_idx, mode)
        if getattr(self.config, "optimizers", None) is None:
            raise RuntimeError(
                "GAN models require named `optimizers` "
                "(`generator` and `discriminator`) in the training config."
            )

        generator_first = bool(self.config.get("generator_first", False))
        turns = (True, False) if generator_first else (False, True)
        for forward_generator in turns:
            step, stats, weight = self._gan_turn(batch, forward_generator)
            if self._check_nan_inf_loss([step.loss], batch_idx):
                return None
            if mode == "train":
                self._run_multi_optimizer_updates([step], stats, weight, batch_idx)
            else:
                self._log_stats(
                    mode,
                    stats,
                    weight,
                    extra_stats={f"{step.name}/loss": step.loss.detach()},
                )
        return None
