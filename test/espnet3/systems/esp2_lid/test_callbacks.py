from pathlib import Path

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from espnet3.systems.esp2_lid.callbacks import BestCheckpointLink


@pytest.mark.parametrize("interrupt", [False, True])
def test_best_checkpoint_is_readable_before_training_ends(tmp_path, interrupt):
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=tmp_path,
        monitor="valid/accuracy",
        mode="max",
        save_top_k=2,
        save_on_train_epoch_end=False,
    )
    seen = []

    class Model(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def training_step(self, batch, batch_idx):
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            self.log("valid/accuracy", [0.1, 0.8, 0.2][self.current_epoch])

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

        def on_train_epoch_end(self):
            link = tmp_path / "valid.accuracy.best.pth"
            assert link.resolve() == Path(checkpoint.best_model_path).resolve()
            saved = torch.load(link, weights_only=True)
            assert saved["epoch"] == min(self.current_epoch, 1)
            assert torch.get_float32_matmul_precision() == "high"
            seen.append(self.current_epoch)
            if interrupt:
                raise RuntimeError("Interrupted after the saved epoch")

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            checkpoint,
            BestCheckpointLink(str(tmp_path)),
        ],
    )
    loader = DataLoader([torch.tensor(1.0)])
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        if interrupt:
            with pytest.raises(RuntimeError, match="Interrupted after"):
                trainer.fit(Model(), loader, loader)
        else:
            trainer.fit(Model(), loader, loader)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    assert seen == ([0] if interrupt else [0, 1, 2])
    assert (tmp_path / "valid.accuracy.best.pth").is_file()
