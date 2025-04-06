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
    model that accepts a `MultilabelAUPRC` object and calls its `update` method
    with predictions and targets.
    For example:
    ```python
    class MyESPnetModel(AbsESPnetModel):
        def update_mAP(self, mAP_function: MultilabelAUPRC):
            ...
            mAP_function.update(predictions, targets)
            ...
    ```
    The model should also have a `get_vocab_size()` function that
    specifies the number of labels/classes.
    """

    def __init__(self):
        super().__init__()
        if torcheval_import_error is not None:
            raise ImportError(
                "`torcheval` is not available. Please install it "
                "via `pip install torcheval` in your environment."
                "More info at: `https://pytorch.org/torcheval/stable/`"
                f"Original error is: {torcheval_import_error}"
            )
        self.mAP_function = None  # init on train start

    def setup_mAP(self, model):
        self.mAP_function = MultilabelAUPRC(num_labels=model.get_vocab_size())

    def on_train_start(self, trainer, pl_module):
        if self.mAP_function is None:
            self.setup_mAP(pl_module.model)
        self.mAP_function.reset()

    def on_validation_start(self, trainer, pl_module):
        if self.mAP_function is None:
            self.setup_mAP(pl_module.model)
        self.mAP_function.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.model.update_mAP(self.mAP_function)
        if not trainer.log_every_n_steps:
            return
        mAP = self.compute_mAP(trainer)
        if mAP is not None:
            pl_module.log(
                "train/mAP", mAP, on_step=True, on_epoch=False, sync_dist=False
            )

    def on_validation_epoch_start(self, trainer, pl_module):
        self.mAP_function.reset()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        pl_module.model.update_mAP(self.mAP_function)  # accumulate

    def on_validation_epoch_end(self, trainer, pl_module):
        mAP = self.compute_mAP(trainer)
        if mAP is not None:
            pl_module.log("valid/epoch_mAP", mAP, on_epoch=True, sync_dist=False)

    def compute_mAP(self, trainer):
        """Computes the mAP."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            mAP = sync_and_compute(self.mAP_function)
        else:
            mAP = self.mAP_function.compute()
        self.mAP_function.reset()
        return mAP
