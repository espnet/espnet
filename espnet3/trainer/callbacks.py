from pathlib import Path
from typing import Any, List, Tuple, Union

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from typeguard import typechecked


@typechecked
class AverageCheckpointsCallback(Callback):
    """
    A custom callback that averages the parameters of the top-K checkpoints
    (based on specified metrics) after training ends, and saves the averaged model.

    Output:
    - The averaged model is saved to `output_dir` with the filename:
      `{monitor_name}.ave_{N}best.pth`

    Notes:
    - Only keys that start with `model.` are averaged (e.g., for models saved
        with `save_weights_only=True`).
    - Parameters with integer types (e.g., `BatchNorm.num_batches_tracked`)
        are not averaged, only accumulated. If other reduction methods are needed
        (e.g., max/min), they should be added explicitly.
    """

    def __init__(self, output_dir, best_ckpt_callbacks):
        self.output_dir = output_dir
        self.best_ckpt_callbacks = best_ckpt_callbacks

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            for ckpt_callback in self.best_ckpt_callbacks:
                checkpoints = list(ckpt_callback.best_k_models.keys())

                avg_state_dict = None
                for ckpt_path in checkpoints:
                    state_dict = torch.load(
                        ckpt_path,
                        map_location="cpu",
                        weights_only=False,
                    )["state_dict"]

                    if avg_state_dict is None:
                        avg_state_dict = state_dict
                    else:
                        for k in avg_state_dict:
                            avg_state_dict[k] = avg_state_dict[k] + state_dict[k]

                for k in avg_state_dict:
                    if str(avg_state_dict[k].dtype).startswith("torch.int"):
                        # For int type, not averaged, but only accumulated.
                        # e.g. BatchNorm.num_batches_tracked
                        # (If there are any cases that requires averaging
                        #  or the other reducing method, e.g. max/min, for integer type,
                        #  please report.)
                        pass
                    else:
                        avg_state_dict[k] = avg_state_dict[k] / len(checkpoints)

                # remove extra prefix in model keys
                new_avg_state_dict = {
                    k.removeprefix("model."): v
                    for k, v in avg_state_dict.items()
                    if k.startswith("model.")
                }

                avg_ckpt_path = (
                    Path(self.output_dir) /
                    (f"{ckpt_callback.monitor.replace('/', '.')}."
                        + f"ave_{len(checkpoints)}best.pth")
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
    """
    Utility function to construct and return a list of standard PyTorch Lightning
    callbacks.

    Included callbacks:
    - ModelCheckpoint for saving the last checkpoint (`save_last`)
    - Multiple ModelCheckpoint callbacks for saving top-K checkpoints based on
        different metrics
    - AverageCheckpointsCallback for saving an averaged model from top-K checkpoints
    - LearningRateMonitor to log learning rate
    - TQDMProgressBar to display training progress

    Args:
        config: Configuration object (e.g., from Hydra or OmegaConf),
            which must include:
            - `expdir`: Output directory for saved models
            - `log_interval`: Refresh rate for the progress bar
            - `best_model_criterion`: List of tuples (metric_name, top_k, mode)
                Example: [("val/wer", 3, "min")] means save top-3 checkpoints
                with lowest val/wer

    Returns:
        List[Callback]: A list of configured PyTorch Lightning callbacks.

    Example:
        >>> from my_callbacks import get_default_callbacks
        >>> callbacks = get_default_callbacks(config)
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
