from espnet3.components.modeling.hf_models import AbsHFTrainingWrapper

from lightning.pytorch.callbacks import Callback


class HFCheckpointSaveCallback(Callback):
    def __init__(self, dirpath: str):
        super().__init__()
        self.dirpath = dirpath

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            if not isinstance(pl_module.model, AbsHFTrainingWrapper):
                raise AttributeError(
                    f"Failed to save Hugging Face model. {pl_module}.model must be an instance of AbsHFTrainingWrapper."
                )
            pl_module.model.save_pretrained(self.dirpath)
