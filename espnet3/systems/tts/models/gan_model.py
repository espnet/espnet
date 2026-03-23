"""GAN-TTS adapter for ESPnet3 Lightning integration."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.modeling.optimization_spec import OptimizationStep


class GANTTSLightningModule(ESPnetLightningModule):
    """Adapt ESPnet2 GAN-TTS model outputs to ESPnet3's named optimizer path."""

    _GAN_STEP_NAMES = {
        0: "generator",
        1: "discriminator",
    }

    def _forward_gan_model(
        self, batch
    ) -> Tuple[List[OptimizationStep], Dict[str, torch.Tensor], torch.Tensor | None]:
        optimizer_steps: List[OptimizationStep] = []
        stats: Dict[str, torch.Tensor] = {}
        weight = None

        for forward_generator in (True, False):
            output = self.model(**batch[1], forward_generator=forward_generator)
            if not isinstance(output, dict):
                raise AssertionError(
                    "GAN-TTS models must return a dict with loss/stats/weight/optim_idx."
                )

            optim_idx = output.get("optim_idx")
            if optim_idx not in self._GAN_STEP_NAMES:
                raise AssertionError(
                    "GAN-TTS models must set optim_idx to 0 (generator) or 1 "
                    f"(discriminator), but got {optim_idx!r}."
                )
            name = self._GAN_STEP_NAMES[optim_idx]

            loss = output.get("loss")
            if not isinstance(loss, torch.Tensor):
                raise AssertionError(
                    f"GAN-TTS '{name}' loss must be a tensor, but got {type(loss)!r}."
                )

            step_stats = output.get("stats")
            if not isinstance(step_stats, dict):
                raise AssertionError(
                    f"GAN-TTS '{name}' stats must be a dict, but got {type(step_stats)!r}."
                )

            optimizer_steps.append(OptimizationStep(loss=loss, name=name))
            stats.update(step_stats)

            step_weight = output.get("weight")
            if weight is None and step_weight is not None:
                weight = step_weight

        return optimizer_steps, stats, weight

    def _step(self, batch, batch_idx, mode):
        if not isinstance(self.model, AbsGANESPnetModel):
            return super()._step(batch, batch_idx, mode)

        optimizer_steps, stats, weight = self._forward_gan_model(batch)

        any_invalid = self._check_nan_inf_loss(
            [step.loss for step in optimizer_steps], batch_idx
        )
        if any_invalid:
            return None

        if mode == "train":
            self._run_multi_optimizer_updates(
                optimizer_steps, stats, weight, batch_idx
            )
        else:
            extra_stats = {
                f"{step.name}/loss": step.loss.detach() for step in optimizer_steps
            }
            self._log_stats(mode, stats, weight, extra_stats=extra_stats)
        return None
