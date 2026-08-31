"""PyTorch lightning callbacks for Hugging Face models."""
from lightning.pytorch.callbacks import Callback

from espnet3.components.modeling.hf_models import AbsHFTrainingWrapper


class HFCheckpointSaveCallback(Callback):
    """Callback for saving HF model checkpoints."""

    def __init__(self, dirpath: str):
        """Initialize the callback."""
        super().__init__()
        self.dirpath = dirpath

    def on_train_end(self, trainer, pl_module):
        """Save the model checkpoint."""
        if trainer.global_rank == 0:
            if not isinstance(pl_module.model, AbsHFTrainingWrapper):
                raise AttributeError(
                    f"""Failed to save Hugging Face model. {pl_module}.model must be an
                    instance of AbsHFTrainingWrapper."""
                )
            pl_module.model.save_pretrained(self.dirpath)
