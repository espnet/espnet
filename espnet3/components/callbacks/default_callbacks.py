"""Callbacks for ESPnet3 trainer."""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Union

import torch
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from typeguard import typechecked

from espnet3.utils.logging_utils import log_component

_LOGGED_CALLBACKS = False


def _metric_to_float(value) -> float | None:
    """Convert logged scalars to `float` and ignore unsupported metric values.

    Tensor values are detached first, and non-scalar inputs return `None` so
    train/valid summary logging can skip them uniformly.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.cpu().item()
    try:
        return float(value)
    except Exception:
        return None


def _format_metrics(metrics: dict[str, float], time_keys: tuple[str, ...]) -> str:
    """Format summary metrics with stable ordering for compact log lines.

    Time keys are emitted first, learning-rate keys last, and the rest are sorted
    alphabetically so train and validation summaries stay easy to scan.
    """
    lr_keys = sorted(k for k in metrics if k.startswith("optim") and "_lr" in k)
    user_keys = sorted(k for k in metrics if k not in time_keys and k not in lr_keys)
    keys = [k for k in time_keys if k in metrics] + user_keys + lr_keys
    return ", ".join(f"{k}={metrics[k]:.6g}" for k in keys)


@typechecked
class AverageCheckpointsCallback(Callback):
    """A custom callback for weight averaging over the top-K checkpoints.

    This can be useful to smooth out fluctuations in weights across the best-performing
    models and can lead to improved generalization performance at inference time.

    Behavior:
        - Loads the state_dict from each of the top-K checkpoints saved by given
          ModelCheckpoint callbacks.
        - Averages the model parameters (keys starting with `model.`).
        - Ignores or simply accumulates integer-type parameters
          (e.g., BatchNorm's `num_batches_tracked`).
        - Saves the averaged model as a `.pth` file in `output_dir`.

    Args:
        output_dir (str or Path):
            The directory where the averaged model will be saved.
        best_ckpt_callbacks (List[ModelCheckpoint]):
            A list of ModelCheckpoint callbacks whose top-K checkpoints will be used
            for averaging. Each callback must have `best_k_models` populated.

    Notes:
        - Only keys that start with `model.` are included in the averaging.
        - The final filename will be:
            `{monitor_name}.ave_{K}best.pth`
        - This callback only runs on the global rank 0 process
            (for distributed training).

    Example:
        >>> avg_ckpt_cb = AverageCheckpointsCallback(
        ...     output_dir="checkpoints/",
        ...     best_ckpt_callbacks=[val_loss_ckpt_cb, acc_ckpt_cb]
        ... )
        >>> trainer = Trainer(callbacks=[avg_ckpt_cb])
    """

    def __init__(self, output_dir, best_ckpt_callbacks):
        """Initialize AverageCheckpointsCallback object."""
        self.output_dir = output_dir
        self.best_ckpt_callbacks = best_ckpt_callbacks

    def on_validation_end(self, trainer, pl_module):
        """At the end of validation, average the top-K checkpoints and save."""
        if trainer.is_global_zero:
            for ckpt_callback in self.best_ckpt_callbacks:
                checkpoints = list(ckpt_callback.best_k_models.keys())
                if not checkpoints:
                    continue

                avg_state_dict = None
                reference_keys = None
                for ckpt_path in checkpoints:
                    state_dict = torch.load(
                        ckpt_path, map_location="cpu", weights_only=False
                    )

                    # for deepspeed checkpoints
                    if "module" in state_dict:
                        state_dict = state_dict["module"]
                    # for PytorchLightning checkpoints
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]

                    if avg_state_dict is None:
                        avg_state_dict = state_dict
                        reference_keys = set(state_dict.keys())
                    else:
                        # Check key consistency
                        current_keys = set(state_dict.keys())
                        if current_keys != reference_keys:
                            raise KeyError(
                                f"Mismatch in keys between checkpoints.\n"
                                f"Expected: {reference_keys}\n"
                                f"Got: {current_keys} (from {ckpt_path})"
                            )
                        for k in avg_state_dict:
                            avg_state_dict[k] = avg_state_dict[k] + state_dict[k]

                for k in avg_state_dict:
                    if str(avg_state_dict[k].dtype).startswith("torch.int"):
                        # For int type, not averaged, but only accumulated.
                        # e.g. BatchNorm.num_batches_tracked
                        # (If there are any cases that requires averaging
                        #  or the other reducing method, e.g. max/min, for integer type,
                        #  please report.)
                        logging.info(
                            "The following parameters were only accumulated, "
                            f"not averaged: {k}"
                        )
                        pass
                    else:
                        avg_state_dict[k] = avg_state_dict[k] / len(checkpoints)

                # remove extra prefix in model keys
                new_avg_state_dict = {
                    k.removeprefix("model."): v
                    for k, v in avg_state_dict.items()
                    if k.startswith("model.")
                }

                avg_ckpt_path = Path(self.output_dir) / (
                    f"{ckpt_callback.monitor.replace('/', '.')}."
                    + f"ave_{len(checkpoints)}best.pth"
                )
                torch.save(new_avg_state_dict, avg_ckpt_path)


@typechecked
class MetricsLogger(Callback):
    """Log compact train and validation metric summaries.

    This callback owns the human-readable metric logging for the default ESPnet3
    training loop. It handles three reporting points in one place:

    - interval-based training batch summaries
    - end-of-epoch training summaries
    - end-of-epoch validation summaries

    The goal is to keep logging responsibility in a single callback instead of
    splitting train and validation reporting across separate callback classes.

    Args:
        log_every_n_steps: Number of training steps between batch-summary log lines.

    Returns:
        None.

    Raises:
        None.

    Notes:
        - Training summaries remove the `train/` prefix from logged metrics.
        - Validation summaries remove the `valid/` prefix from logged metrics.
        - Validation sanity-check runs are ignored to avoid noisy startup logs.

    Examples:
        >>> cb = MetricsLogger(log_every_n_steps=200)
        >>> trainer = Trainer(callbacks=[cb, ...])

        Example train log output:
        `20epoch:train:4201-4400batch: iter_time=6.212e-05, loss=46.669`

        Example validation log output:
        `epoch_summary:20epoch:valid: valid_time=1.42, acc=0.91, loss=0.83`
    """

    def __init__(self, log_every_n_steps: int = 500):
        """Initialize the logger with a reporting interval."""
        self.log_every_n_steps = int(log_every_n_steps)
        self._sum = defaultdict(float)
        self._count = 0
        self._start_batch = None
        self._epoch_sum = defaultdict(float)
        self._epoch_count = 0
        self._last_batch_end_time = None
        self._batch_start_time = None
        self._forward_start_time = None
        self._backward_start_time = None
        self._optim_step_start_time = None
        self._validation_start_time = None

    def _reset(self):
        """Clear buffered metrics for the next logging window."""
        self._sum.clear()
        self._count = 0
        self._start_batch = None
        self._batch_start_time = None
        self._forward_start_time = None
        self._backward_start_time = None
        self._optim_step_start_time = None

    def _reset_epoch(self):
        """Clear epoch-level buffered metrics."""
        self._epoch_sum.clear()
        self._epoch_count = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Start timers for a training batch."""
        # Use wall clock for simplicity; Lightning doesn't expose loader timing.
        t = time.perf_counter()
        if self._last_batch_end_time is not None:
            iter_time = t - self._last_batch_end_time
            self._sum["iter_time"] += iter_time
        self._batch_start_time = t
        self._forward_start_time = t

    def on_before_backward(self, trainer, pl_module, loss):
        """Mark the end of forward pass and start backward timing."""
        t = time.perf_counter()
        if self._forward_start_time is not None:
            self._sum["forward_time"] += t - self._forward_start_time
        self._backward_start_time = t

    def on_after_backward(self, trainer, pl_module):
        """Record backward time."""
        t = time.perf_counter()
        if self._backward_start_time is not None:
            self._sum["backward_time"] += t - self._backward_start_time
        self._backward_start_time = None

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Start optimizer step timing."""
        self._optim_step_start_time = time.perf_counter()

    def on_after_optimizer_step(self, trainer, pl_module):
        """Record optimizer step time."""
        if self._optim_step_start_time is not None:
            self._sum["optim_step_time"] += (
                time.perf_counter() - self._optim_step_start_time
            )
        self._optim_step_start_time = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Aggregate metrics and emit a summary line when due."""
        metrics = trainer.callback_metrics or {}

        if self._start_batch is None:
            self._start_batch = batch_idx + 1

        for key, value in metrics.items():
            if key in {"epoch", "step"} or str(key).startswith("valid/"):
                continue
            key = str(key)
            if key.startswith("train/"):
                key = key[len("train/") :]
            value = _metric_to_float(value)
            if value is None:
                continue
            self._sum[key] += value
            self._epoch_sum[key] += value

        # Optimizer learning rates
        try:
            optimizers = trainer.optimizers or []
        except Exception:
            optimizers = []
        for opt_idx, optim in enumerate(optimizers):
            for group_idx, group in enumerate(getattr(optim, "param_groups", [])):
                lr = group.get("lr")
                if lr is not None:
                    self._sum[f"optim{opt_idx}_lr{group_idx}"] += float(lr)
                    self._epoch_sum[f"optim{opt_idx}_lr{group_idx}"] += float(lr)

        if self._batch_start_time is not None:
            self._sum["train_time"] += time.perf_counter() - self._batch_start_time
            self._epoch_sum["train_time"] += (
                time.perf_counter() - self._batch_start_time
            )

        self._last_batch_end_time = time.perf_counter()
        self._count += 1
        self._epoch_count += 1
        if self.log_every_n_steps <= 0:
            return
        if (batch_idx + 1) % self.log_every_n_steps != 0:
            return

        avg = {k: (v / self._count if self._count else v) for k, v in self._sum.items()}
        metrics_str = _format_metrics(
            avg,
            (
                "iter_time",
                "forward_time",
                "backward_time",
                "optim_step_time",
                "train_time",
            ),
        )
        start = self._start_batch or (batch_idx + 1)
        end = batch_idx + 1
        epoch = trainer.current_epoch + 1
        logging.info("%depoch:train:%d-%dbatch: %s", epoch, start, end, metrics_str)
        self._reset()

    def on_train_epoch_end(self, trainer, pl_module):
        """Reset buffers at the end of each training epoch."""
        if self._epoch_count > 0:
            avg = {
                k: (v / self._epoch_count if self._epoch_count else v)
                for k, v in self._epoch_sum.items()
            }
            metrics_str = _format_metrics(
                avg,
                (
                    "iter_time",
                    "forward_time",
                    "backward_time",
                    "optim_step_time",
                    "train_time",
                ),
            )
            epoch = trainer.current_epoch + 1
            logging.info("epoch_summary:%depoch:train: %s", epoch, metrics_str)
        self._reset()
        self._reset_epoch()

    def on_validation_epoch_start(self, trainer, pl_module):
        """Start wall-clock timing for one validation epoch."""
        if getattr(trainer, "sanity_checking", False):
            return
        self._validation_start_time = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        """Emit one log line with aggregated validation metrics."""
        if getattr(trainer, "sanity_checking", False):
            return

        metrics = {}
        callback_metrics = trainer.callback_metrics or {}
        for key, value in callback_metrics.items():
            key = str(key)
            if key in {"epoch", "step"} or not key.startswith("valid/"):
                continue
            converted = _metric_to_float(value)
            if converted is None:
                continue
            metrics[key[len("valid/") :]] = converted

        if self._validation_start_time is not None:
            metrics["valid_time"] = time.perf_counter() - self._validation_start_time
        self._validation_start_time = None

        if not metrics:
            return

        epoch = trainer.current_epoch + 1
        logging.info(
            "epoch_summary:%depoch:valid: %s",
            epoch,
            _format_metrics(metrics, ("valid_time",)),
        )


@typechecked
def get_default_callbacks(
    exp_dir: str = "./exp",
    log_interval: int = 500,
    best_model_criterion: Union[List[Tuple[str, int, str]], List[List]] = [
        ("valid/loss", 3, "min")
    ],
) -> List[Callback]:
    """Return a list of callbacks tailored for most training workflows.

    Includes:
        - `ModelCheckpoint` for saving the last model checkpoint (`save_last`)
        - One or more `ModelCheckpoint`s for saving the top-K checkpoints according to
            specific metrics
        - `AverageCheckpointsCallback` to compute and save the average model from top-K
            checkpoints
        - `LearningRateMonitor` to track and log learning rates during training
        - `MetricsLogger` to emit train and validation summaries
        - `TQDMProgressBar` to show a rich progress bar during training

    Args:
        exp_dir (str): Directory to store checkpoints and logs.
        log_interval (int): Frequency (in training steps) to refresh the progress bar.
        best_model_criterion (List[Tuple[str, int, str]]): A list of criteria for
            saving top-K checkpoints.
            Each item is a tuple: (name, top_k, mode), where:
            - `name` (str): The name of the validation value to monitor
                (e.g., "val/loss").
            - `top_k` (int): Number of best models to keep.
            - `mode` (str): "min" to keep models with lowest value, "max" for highest.

    Returns:
        List[Callback]: A list of callbacks to be passed to the PyTorch Lightning
            Trainer.

    Example:
        >>> from default_callbacks import get_default_callbacks
        >>> callbacks = get_default_callbacks(
        ...     exp_dir="./exp",
        ...     log_interval=100,
        ...     best_model_criterion=[("val/loss", 5, "min"), ("val/acc", 3, "max")]
        ... )
        >>> trainer = Trainer(callbacks=callbacks, ...)
    """
    last_ckpt_callback = ModelCheckpoint(
        dirpath=exp_dir,
        save_last="link",
        filename="step{step}",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
        save_weights_only=False,
    )

    best_ckpt_callbacks = []
    for monitor, nbest, mode in best_model_criterion:
        best_ckpt_callbacks.append(
            ModelCheckpoint(
                save_top_k=nbest,
                monitor=monitor,
                mode=mode,  # "min" or "max"
                dirpath=exp_dir,
                save_last=False,
                # Add monitor to filename to avoid overwriting
                # when multiple metrics are used
                filename="epoch{epoch}_step{step}_" + monitor.replace("/", "."),
                auto_insert_metric_name=False,
                save_on_train_epoch_end=False,
                save_weights_only=True,
                enable_version_counter=False,  # just overwrite
            )
        )
    ave_ckpt_callback = AverageCheckpointsCallback(
        output_dir=exp_dir, best_ckpt_callbacks=best_ckpt_callbacks
    )

    # Monitor learning rate
    lr_callback = LearningRateMonitor()

    # Progress bar
    progress_bar_callback = TQDMProgressBar(refresh_rate=log_interval)

    callbacks = [
        last_ckpt_callback,
        *best_ckpt_callbacks,  # unpack list to add them to the list of callbacks.
        ave_ckpt_callback,
        lr_callback,
        MetricsLogger(log_every_n_steps=log_interval),
        progress_bar_callback,
    ]
    global _LOGGED_CALLBACKS
    if not _LOGGED_CALLBACKS:
        _LOGGED_CALLBACKS = True
        for idx, cb in enumerate(callbacks):
            log_component(
                logging.getLogger(__name__),
                kind="Callback",
                label=str(idx),
                obj=cb,
                max_depth=2,
            )
    return callbacks
