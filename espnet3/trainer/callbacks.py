"""Callbacks for ESPnet3 trainer."""

import logging
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
def get_default_callbacks(
    expdir: str = "./exp",
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
        - `TQDMProgressBar` to show a rich progress bar during training

    Args:
        expdir (str): Directory to store checkpoints and logs.
        log_interval (int): Frequency (in training steps) to refresh the progress bar.
        best_model_criterion (List[Tuple[str, int, str]]): A list of criteria for
            saving top-K checkpoints.
            Each item is a tuple: (metric_name, top_k, mode), where:
            - `metric_name` (str): The name of the validation metric to monitor
                (e.g., "val/loss").
            - `top_k` (int): Number of best models to keep.
            - `mode` (str): "min" to keep models with lowest metric, "max" for highest.

    Returns:
        List[Callback]: A list of callbacks to be passed to the PyTorch Lightning
            Trainer.

    Example:
        >>> from callbacks import get_default_callbacks
        >>> callbacks = get_default_callbacks(
        ...     expdir="./exp",
        ...     log_interval=100,
        ...     best_model_criterion=[("val/loss", 5, "min"), ("val/acc", 3, "max")]
        ... )
        >>> trainer = Trainer(callbacks=callbacks, ...)
    """
    last_ckpt_callback = ModelCheckpoint(
        dirpath=expdir,
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
                dirpath=expdir,
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
        output_dir=expdir, best_ckpt_callbacks=best_ckpt_callbacks
    )

    # Monitor learning rate
    lr_callback = LearningRateMonitor()

    # Progress bar
    progress_bar_callback = TQDMProgressBar(refresh_rate=log_interval)

    return [
        last_ckpt_callback,
        *best_ckpt_callbacks,  # unpack list to add them to the list of callbacks.
        ave_ckpt_callback,
        lr_callback,
        progress_bar_callback,
    ]
