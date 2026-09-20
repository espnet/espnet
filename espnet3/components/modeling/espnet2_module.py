"""Opt-in ESPnet2 single-optimizer semantics for Lightning ASR recipes."""

import logging

import torch
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel

from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet3.components.modeling.lightning_module import ESPnetLightningModule

logger = logging.getLogger(__name__)


class CPUCTCLoss(torch.nn.Module):
    """Evaluate only the CTC loss on CPU while retaining cross-device autograd.

    Args:
        loss: Existing built-in CTCLoss, including its reduction and blank settings.

    Notes:
        The projection and log-softmax remain on their original device. This
        wrapper has no parameters, so model checkpoint keys remain unchanged.
    """

    def __init__(self, loss):
        """Retain the original loss implementation and its options."""
        super().__init__()
        self.loss = loss

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """Compute the CPU loss and return it to the input device.

        Args:
            log_probs: Time-major log probabilities, with autograd attached.
            targets: Concatenated target token IDs.
            input_lengths: Encoder lengths for each utterance.
            target_lengths: Target lengths for each utterance.

        Returns:
            Loss with the original reduction, on ``log_probs.device``.
        """
        return self.loss(
            log_probs.cpu(), targets.cpu(), input_lengths.cpu(), target_lengths.cpu()
        ).to(log_probs.device)


class ESPnet2LightningModule(ESPnetLightningModule):
    """Preserve native single-optimizer updates inside the ESPnet3 pipeline.

    Args:
        model: ESPnet model returning ``loss, stats, weight``.
        config: Training config with ``espnet2_compat.accum_grad`` (default 1),
            ``grad_clip`` (5), ``grad_clip_type`` (2), and ``ctc_on_cpu`` (false).
            Use trainer accumulation 1 and trainer gradient clipping 0; this
            module owns both operations. ``seed`` is reset to seed + epoch + 1.

    Raises:
        ValueError: Multiple optimizers, conflicting trainer settings, or an
            unsupported CTC implementation is requested.

    Notes:
        Like espnet2.train.trainer.Trainer, incomplete epoch tails are retained
        until the next epoch's first complete accumulation group. Resume starts
        without residual gradients, matching native checkpoints. Lightning still
        owns device placement, AMP scaling, distributed backward and checkpoints.

    Examples:
        Set ``lightning_module`` to this class's dotted path and configure
        ``espnet2_compat: {accum_grad: 4, grad_clip: 5.0}`` in a recipe YAML.
    """

    def __init__(self, model, config):
        """Validate the compatibility contract and optionally wrap CTC."""
        super().__init__(model, config)
        options = config.espnet2_compat
        self.accum_grad = int(options.get("accum_grad", 1))
        self.grad_clip = float(options.get("grad_clip", 5.0))
        self.grad_clip_type = float(options.get("grad_clip_type", 2.0))
        if config.get("optimizers") or self.accum_grad < 1:
            raise ValueError(
                "ESPnet2 compatibility requires one optimizer and accum_grad >= 1"
            )
        if config.trainer.get("accumulate_grad_batches", 1) != 1:
            raise ValueError("Use espnet2_compat.accum_grad, not trainer accumulation")
        if config.trainer.get("gradient_clip_val", 0) not in (None, 0, 0.0):
            raise ValueError("Use espnet2_compat.grad_clip, not trainer clipping")
        if options.get("ctc_on_cpu", False):
            if not hasattr(model, "ctc") or model.ctc.ctc_type != "builtin":
                raise ValueError("CPU CTC requires the built-in ESPnet CTC loss")
            model.ctc.ctc_loss = CPUCTCLoss(model.ctc.ctc_loss)
        self.automatic_optimization = False
        self._finite_gradient = True
        self._epoch_has_update = False

    def configure_optimizers(self):
        """Build the source optimizer and scheduler for manual update control.

        Returns:
            Optimizer and scheduler lists for Lightning checkpoint persistence.
        """
        optimizer = instantiate(self.config.optimizer, params=self.model.parameters())
        scheduler = instantiate(self.config.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        """Reset model RNG to the native one-based epoch seed."""
        set_all_random_seed(int(self.config.seed) + self.current_epoch + 1)
        self._epoch_has_update = False

    def _restore_ddp_buffer_sync(self):
        """Restore native DDP forward synchronization after manual backward.

        Lightning suppresses backward synchronization inside manual forward,
        which also clears DDP's next-forward buffer synchronization flag.
        Let DDP perform its own broadcast, including noncontiguous buffers and
        process groups, instead of copying buffers outside its forward path.
        """
        trainer = getattr(self, "_trainer", None)
        wrapper = trainer.strategy.model if trainer is not None else None
        if isinstance(wrapper, DistributedDataParallel):
            wrapper.require_forward_param_sync = True

    def on_train_batch_start(self, batch, batch_idx):
        """Restore the native buffer broadcast before each training forward.

        Args:
            batch: Utterance IDs and collated model inputs.
            batch_idx: Zero-based batch position within the current epoch.
        """
        self._restore_ddp_buffer_sync()

    def on_validation_start(self):
        """Broadcast training buffers on the first validation forward, as in DDP."""
        self._restore_ddp_buffer_sync()

    def _log_stats(self, mode, stats, weight, extra_stats=None):
        """Accumulate native globally reduced statistics in double precision.

        ESPnet2's reporter converts scalars to Python float before weighted
        epoch accumulation. Passing float64 tensors keeps Lightning from
        rounding that accumulation to float32. Per-step DDP reduction already
        occurs in training_step/validation_step, so no second rank reduction
        is needed here.
        """
        if getattr(self, "_trainer", None) is None:
            return
        values = dict(stats)
        if extra_stats is not None:
            values.update(extra_stats)
        values = {
            f"{mode}/{name}": torch.as_tensor(
                value, dtype=torch.float64, device=self.device
            ).detach()
            for name, value in values.items()
            if value is not None
        }
        self.log_dict(
            values,
            prog_bar=True,
            logger=True,
            on_step=mode == "train",
            on_epoch=True,
            sync_dist=False,
            batch_size=int(weight.item() if hasattr(weight, "item") else weight),
        )

    def on_validation_model_zero_grad(self):
        """Keep incomplete accumulation groups across validation as ESPnet2 does."""
        # Lightning otherwise clears the last training batch's residual gradients.
        pass

    def _clip_gradients(self):
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip, self.grad_clip_type
        )
        self._finite_gradient = bool(torch.isfinite(norm))
        if not self._finite_gradient:
            logger.warning("Nonfinite gradient norm; skipping optimizer update")

    def on_before_optimizer_step(self, optimizer):
        """Clip AMP gradients after Lightning unscales them.

        Args:
            optimizer: Raw optimizer provided by Lightning's precision plugin.
        """
        if getattr(self.trainer.precision_plugin, "scaler", None) is not None:
            self._clip_gradients()

    def _forward_batch(self, batch):
        """Match native AMP's BF16 preference while retaining GradScaler.

        ESPnet2 enables GradScaler for AMP even when autocast selects BF16.
        Lightning's 16-mixed supplies that scaler; override only the model's
        autocast dtype on BF16-capable CUDA devices. FP32 calls are unchanged.
        """
        if torch.is_autocast_enabled("cuda") and torch.cuda.is_bf16_supported():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.model(**batch[1])
        return self.model(**batch[1])

    def training_step(self, batch, batch_idx):
        """Accumulate and update using the native epoch-local step boundary.

        Args:
            batch: Utterance IDs and tensor dictionary from the ESPnet collator.
            batch_idx: Zero-based batch position within the current epoch.

        Returns:
            Detached loss for Lightning diagnostics; backward is already complete.
        """
        loss, stats, weight = self._forward_batch(batch)
        if torch.distributed.is_initialized():
            loss = (loss * weight.to(loss.dtype)).sum()
            stats, weight = recursive_average(stats, weight, True)
            loss = loss / weight * torch.distributed.get_world_size()
        self._log_stats("train", stats, weight)
        # Native Trainer exits autocast before backward and optimizer updates.
        # Lightning's manual optimization keeps its outer autocast active.
        with torch.autocast("cuda", enabled=False):
            self.manual_backward(loss / self.accum_grad)
            if (batch_idx + 1) % self.accum_grad == 0:
                optimizer = self.optimizers()
                scaler = getattr(self.trainer.precision_plugin, "scaler", None)
                if scaler is None:
                    self._clip_gradients()
                if scaler is not None or self._finite_gradient:
                    optimizer.step()
                if self._finite_gradient:
                    self._epoch_has_update = True
                    if self.config.get("scheduler_interval", "step") == "step":
                        self.lr_schedulers().step()
                optimizer.zero_grad()
        return loss.detach()

    def validation_step(self, batch, batch_idx):
        """Report validation statistics using the native distributed weighting.

        Args:
            batch: Utterance IDs and collated model inputs.
            batch_idx: Zero-based validation position.

        Returns:
            Unmodified local loss for diagnostics.
        """
        loss, stats, weight = self._forward_batch(batch)
        if torch.distributed.is_initialized():
            stats, weight = recursive_average(stats, weight, True)
        self._log_stats("valid", stats, weight)
        return loss

    def on_train_epoch_end(self):
        """Update epoch schedulers after validation, retaining incomplete gradients."""
        if self.config.get("scheduler_interval", "step") == "epoch":
            scheduler = self.lr_schedulers()
            monitor = self.config.get("scheduler_monitor")
            if monitor:
                scheduler.step(self.trainer.callback_metrics[monitor])
            else:
                scheduler.step()
        if not self._epoch_has_update:
            logger.warning(
                "No valid optimizer update in this epoch; stopping as ESPnet2 does"
            )
            self.trainer.should_stop = True
