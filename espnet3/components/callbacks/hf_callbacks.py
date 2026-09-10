"""PyTorch lightning callbacks for Hugging Face models."""

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from espnet3.components.modeling.hf_models import AbsHFTrainingWrapper


class HFCheckpointSaveCallback(Callback):
    """Callback for saving HF model checkpoints."""

    def __init__(self, dirpath: str):
        """Initialize the callback.

        Args:
            dirpath (str): Location where checkpoints will be saved.
        """
        super().__init__()
        self.dirpath = dirpath

    def _check_module(self, pl_module):
        model = getattr(pl_module, "model", None)
        if not isinstance(model, AbsHFTrainingWrapper):
            raise TypeError(f"""Failed to save Hugging Face model.
                {type(pl_module).__name__}.model must be an instance
                of AbsHFTrainingWrapper, got {type(model).__name__}""")

    def on_train_start(self, trainer, pl_module):
        self._check_module(pl_module)

    def on_train_end(self, trainer, pl_module):
        """Save the model checkpoint."""
        if trainer.global_rank == 0:
            self._check_module(pl_module)

            # Restoring the best checkpoint
            checkpoint_callback = next(
                (
                    callback
                    for callback in trainer.callbacks
                    if isinstance(callback, ModelCheckpoint)
                ),
                None,
            )

            if checkpoint_callback is None:
                raise RuntimeError("Could not find a ModelCheckpoint callback.")

            best_model_path = checkpoint_callback.best_model_path

            if not best_model_path:
                raise RuntimeError("ModelCheckpoint has no best_model_path.")

            checkpoint = torch.load(best_model_path, map_location=pl_module.device)

            pl_module.load_state_dict(checkpoint["state_dict"])

            pl_module.model.save_pretrained(self.dirpath)
