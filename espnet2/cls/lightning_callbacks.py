import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback

try:
    from torcheval.metrics import MultilabelAUPRC
    from torcheval.metrics.toolkit import sync_and_compute

    torcheval_import_error = None
except ImportError as err:
    torcheval_import_error = err


class MultilabelAUPRCCallback(Callback):
    """Computes and logs Multilabel AUPRC (mAP) at the end of each validation epoch.

    To use this callback, you must implement a `update_mAP` method in the espnet
    model wrapped inside your LightningModule that accepts a `MultilabelAUPRC` object
     and calls its `update` method with predictions
    and targets. For example:
    ```python
    class MyLightningModule(pl.LightningModule):
        def update_mAP(self, mAP_computer):
            ...
            mAP_computer.update(predictions, targets)
    ```
    The model should also have a `get_vocab_size()` function that
    specifies the number of classes.
    """

    def __init__(self):
        super().__init__()
        if torcheval_import_error is not None:
            raise torcheval_import_error
        self.mAP_computer = None  # init on train start

    def on_train_start(self, trainer, pl_module):
        if self.mAP_computer is None:
            self.mAP_computer = MultilabelAUPRC(pl_module.model.get_vocab_size())
        self.mAP_computer.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.model.update_mAP(self.mAP_computer)
        if not trainer.log_every_n_steps:
            return
        if trainer.is_global_zero:
            mAP = self.compute_mAP()
            pl_module.log(
                "train/mAP", mAP, on_step=True, on_epoch=False, sync_dist=True
            )

    def on_validation_epoch_start(self, trainer, pl_module):
        self.mAP_computer.reset()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        pl_module.model.update_mAP(self.mAP_computer)  # accumulate

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            mAP = self.compute_mAP()
            pl_module.log("val/mAP", mAP, on_epoch=True, sync_dist=True)

    def compute_mAP(self):
        """Computes the mAP."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            mAP = sync_and_compute(self.mAP_computer)
        else:
            mAP = self.mAP_computer.compute()
        self.mAP_computer.reset()
        return mAP
