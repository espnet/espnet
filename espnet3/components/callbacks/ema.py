"""Callback for Exponential Moving Average (EMA) of model weights."""

from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.distributed as dist

from espnet3.components.callbacks.vendored_ema import EMA


class EMACallback(pl.Callback):
    """
    ESPnet3's EMA callback system.

    - Updates once per true optimizer step (not per micro-step).
    - Swaps EMA weights in for validation/test, restores afterward.
    - Saves under 'ema_model_state_dict'.
    """

    def __init__(self, decay: float = 0.9999, **ema_kwargs):
        """Configure the callback.

        Args:
            decay: EMA decay rate, forwarded to ``EMA`` as ``beta``.
            **ema_kwargs: Extra keyword arguments forwarded to ``EMA``.
        """
        self.decay = decay
        self.ema_kwargs = ema_kwargs
        self.ema: EMA | None = None
        self._backup: dict | None = None
        self._last_global_step: int = 0

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        """Create the EMA copy of the model at the start of ``fit``."""
        # Initialize EMA on the main process only
        if stage == "fit" and trainer.is_global_zero:
            self.ema = EMA(
                pl_module.model,
                beta=self.decay,
                include_online_model=False,
                **self.ema_kwargs,
            ).to(pl_module.device)

    def on_train_start(self, trainer, pl_module):
        """Record the step counter training actually starts from."""
        # Runs after any checkpoint restore, so on resume this picks up the
        # restored global_step instead of replaying pre-resume steps as updates.
        self._last_global_step = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update the EMA weights once per true optimizer step."""
        # Update EMA once per true optimizer step
        if self.ema is None or not trainer.is_global_zero:
            return
        if trainer.global_step == self._last_global_step:
            return  # mid-accumulation micro-step, no real optimizer step
        self._last_global_step = trainer.global_step
        self.ema.update()

    def _swap_in_ema(self, trainer, pl_module):
        """Load the EMA weights into the online model for evaluation.

        ``ema_pytorch.EMA`` keeps the averaged weights in ``self.ema.ema_model``
        (a copy of the online model) and has no torch-ema-style store/restore, so
        we back up the online weights and load ``ema_model``'s state.

        EMA lives only on the main rank (``self.ema`` is None elsewhere). On
        multi-GPU we therefore broadcast the main rank's EMA weights to every rank
        so all ranks validate on the same EMA weights (otherwise non-main ranks
        would validate on the online model and pollute the aggregated metric).
        """
        distributed = (
            trainer.world_size > 1 and dist.is_available() and dist.is_initialized()
        )

        # Only the main rank knows whether EMA exists; agree across ranks so all
        # ranks take the same path (and issue the same collectives).
        has_ema = self.ema is not None
        if distributed:
            flag = torch.tensor([int(has_ema)], device=pl_module.device)
            dist.broadcast(flag, src=0)
            has_ema = bool(flag.item())
        if not has_ema:
            return

        self._backup = {
            k: v.detach().clone() for k, v in pl_module.model.state_dict().items()
        }
        if self.ema is not None:  # main rank: put EMA weights into the online model
            pl_module.model.load_state_dict(self.ema.ema_model.state_dict())
        # send main's (EMA) weights to every rank, in registration order
        if distributed:
            for tensor in pl_module.model.state_dict().values():
                if torch.is_tensor(tensor):
                    dist.broadcast(tensor, src=0)

    def _restore_online(self, pl_module):
        """Restore the online weights saved by :meth:`_swap_in_ema`.

        Restore is keyed on ``self._backup`` (set on every rank that swapped), not
        on ``self.ema`` — non-main ranks have ``self.ema is None`` but still hold a
        backup, and must restore to avoid leaving EMA weights in the online model
        (which would desync DDP when training resumes).
        """
        if self._backup is None:
            return
        pl_module.model.load_state_dict(self._backup)
        self._backup = None

    def on_validation_start(self, trainer, pl_module):
        """Swap the EMA weights in so validation runs on them."""
        self._swap_in_ema(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        """Put the online weights back after validation."""
        self._restore_online(pl_module)

    def on_test_start(self, trainer, pl_module):
        """Swap the EMA weights in so testing runs on them."""
        self._swap_in_ema(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        """Put the online weights back after testing."""
        self._restore_online(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Store the EMA state under ``ema_model_state_dict``."""
        if trainer.is_global_zero and self.ema is not None:
            checkpoint["ema_model_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Restore the EMA state from ``ema_model_state_dict`` if present."""
        if "ema_model_state_dict" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema_model_state_dict"])
