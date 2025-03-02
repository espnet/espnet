from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback

from espnet2.cls.lightning_callbacks import MultilabelAUPRCCallback

user_callback_choices = {
    "mAP_logging": MultilabelAUPRCCallback,
}


class AverageCheckpointsCallback(Callback):
    def __init__(self, output_dir, best_ckpt_callbacks):
        self.output_dir = output_dir
        self.best_ckpt_callbacks = best_ckpt_callbacks

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            for ckpt_callback in self.best_ckpt_callbacks:
                checkpoints = list(ckpt_callback.best_k_models.keys())

                avg_state_dict = None
                for ckpt_path in checkpoints:
                    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

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

                avg_ckpt_path = Path(self.output_dir) / (
                    ckpt_callback.monitor.replace("/", ".")
                    + f".ave_{len(checkpoints)}best.pth"
                )
                torch.save(new_avg_state_dict, avg_ckpt_path)
