"""Lightning module running ESPnet2 GAN models in generator/discriminator turns."""

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.modeling.optimization_spec import OptimizationStep


class GANLightningModule(ESPnetLightningModule):
    """Train an ``AbsGANESPnetModel`` with ESPnet3's named optimizers.

    ESPnet2 GAN models (VISinger, VITS, JETS, ...) are called once per turn
    with ``forward_generator`` and return a dict with ``loss``, ``stats`` and
    ``weight``. As in ``espnet2.train.gan_trainer.GANTrainer``, every batch
    runs a generator turn and a discriminator turn, discriminator first unless
    ``generator_first`` is true. Each turn updates the optimizer of the same
    name, so ``config.optimizers`` must define ``generator`` and
    ``discriminator``. Models that are not GAN models use the base step.

    Example:
        .. code-block:: yaml

            generator_first: true
            optimizers:
              generator:
                optimizer: {_target_: torch.optim.AdamW, lr: 2.0e-4}
                params: generator
              discriminator:
                optimizer: {_target_: torch.optim.AdamW, lr: 2.0e-4}
                params: discriminator
    """

    def _step(self, batch, batch_idx, mode):
        """Run the generator and discriminator turns for one batch."""
        if not isinstance(self.model, AbsGANESPnetModel):
            return super()._step(batch, batch_idx, mode)
        if self.config.get("optimizers") is None:
            raise ValueError(
                "GAN models need `optimizers.generator` and "
                "`optimizers.discriminator` in the training config."
            )

        if self.config.get("generator_first", False):
            turns = ["generator", "discriminator"]
        else:
            turns = ["discriminator", "generator"]
        for turn in turns:
            output = self.model(**batch[1], forward_generator=turn == "generator")
            if not isinstance(output, dict):
                raise AssertionError(
                    f"GAN models must return a dict, but got {type(output)}."
                )
            step = OptimizationStep(loss=output["loss"], name=turn)
            if self._check_nan_inf_loss([step.loss], batch_idx):
                return None
            if mode == "train":
                self._run_multi_optimizer_updates(
                    [step], output["stats"], output["weight"], batch_idx
                )
            else:
                self._log_stats(
                    mode,
                    output["stats"],
                    output["weight"],
                    extra_stats={f"{turn}/loss": step.loss.detach()},
                )
        return None
