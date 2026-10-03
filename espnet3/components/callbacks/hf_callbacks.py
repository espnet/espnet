"""PyTorch lightning callbacks for Hugging Face models."""

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from espnet3.systems.asr.models.hf_models import AbsHFTrainingWrapper


class HFCheckpointSaveCallback(Callback):
    """Callback for saving HF models.

    Since ESPnet3 uses LightningModules, model checkpoints will be saved in
    Lightning's format. This format is incompatible with how Hugging Face
    models are saved. To address this, this callback saves the best model
    in the format expected by HF at the end of training.
    This way, the model can be loaded natively using only `transformers`
    (no ESPnet). This is especially important if someone wants to fine-tune
    a model in ESPnet and publish it on HF.

    As such, this callback should only be used with models derived from
    AbsHFTrainingWrapper (a compatibility wrapper for training HF models
    in ESPnet). This condition is checked at the beginning of training,
    and an expection is thrown if the model is not compatible.
    """

    def __init__(self, dirpath: str):
        """Initialize the callback.

        Args:
            dirpath (str): Location where checkpoints will be saved.
        """
        super().__init__()
        self.dirpath = dirpath

    def _check_module(self, pl_module):
        """Check that the model is actually a HF model."""
        model = getattr(pl_module, "model", None)
        if not isinstance(model, AbsHFTrainingWrapper):
            raise TypeError(f"""Failed to save Hugging Face model.
                {type(pl_module).__name__}.model must be an instance
                of AbsHFTrainingWrapper, got {type(model).__name__}""")

    def on_train_start(self, trainer, pl_module):
        """Check that model is valid before training starts."""
        self._check_module(pl_module)

    def on_train_end(self, trainer, pl_module):
        """Save the best model."""
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
