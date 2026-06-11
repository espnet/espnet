"""Helpers for working with Hugging Face models."""

from lightning.pytorch.callbacks import Callback


class HFCheckpointSaveCallback(Callback):
    def __init__(self, dirpath: str):
        super().__init__()
        self.dirpath = dirpath

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        hf_model = getattr(pl_module.model, "model", None)
        if hf_model is None:
            raise AttributeError(
                f"Expected {hf_model}.model to be a Hugging Face model."
            )

        if not hasattr(hf_model, "save_pretrained"):
            raise AttributeError(
                f"Couldn't save Hugging Face model: {hf_model} has no method `save_pretrained()`"
            )

        hf_model.save_pretrained(self.dirpath)
