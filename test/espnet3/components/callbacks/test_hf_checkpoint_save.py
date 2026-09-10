from unittest.mock import MagicMock, patch

import pytest
import torch

from lightning.pytorch.callbacks import ModelCheckpoint

from espnet3.components.callbacks.hf_callbacks import HFCheckpointSaveCallback
from espnet3.components.modeling.hf_models import AbsHFTrainingWrapper


class TestHFWrapper(AbsHFTrainingWrapper):
    """Wrapper for testing functionality without downloading models from HF."""

    def collect_feats(self, **batch):
        pass


def test_hf_checkpoint_save_on_train_end(tmp_path):
    model = TestHFWrapper.__new__(TestHFWrapper)
    model.save_pretrained = MagicMock()

    pl_module = MagicMock()
    pl_module.model = model
    pl_module.device = torch.device("cpu")

    best_model_path = str(tmp_path / "best.ckpt")

    checkpoint_callback = MagicMock(spec=ModelCheckpoint)
    checkpoint_callback.best_model_path = best_model_path

    trainer = MagicMock()
    trainer.global_rank = 0
    trainer.callbacks = [checkpoint_callback]

    checkpoint = {
        "state_dict": {
            "model.weight": torch.tensor([1.0]),
        }
    }

    callback = HFCheckpointSaveCallback(
        dirpath=str(tmp_path / "hf_model")
    )

    with patch("torch.load", return_value=checkpoint) as mock_load:
        callback.on_train_end(trainer, pl_module)

    mock_load.assert_called_once_with(
        best_model_path,
        map_location=pl_module.device,
    )

    pl_module.load_state_dict.assert_called_once_with(
        checkpoint["state_dict"]
    )

    model.save_pretrained.assert_called_once_with(
        str(tmp_path / "hf_model")
    )


def test_hf_checkpoint_save_invalid_model_class(tmp_path):
    """Test that callback only accepts HF training wrappers."""
    callback = HFCheckpointSaveCallback(str(tmp_path))

    pl_module = MagicMock()
    pl_module.model = MagicMock()

    trainer = MagicMock()
    trainer.global_rank = 0

    with pytest.raises(TypeError):
        callback.on_train_start(trainer, pl_module)

    with pytest.raises(TypeError):
        callback.on_train_end(trainer, pl_module)


def test_hf_checkpoint_save_on_non_global_zero(tmp_path):
    """Test that callback is skipped if global rank is not zero."""
    callback = HFCheckpointSaveCallback(str(tmp_path))

    model = TestHFWrapper.__new__(TestHFWrapper)
    model.save_pretrained = MagicMock()

    pl_module = MagicMock()
    pl_module.model = model

    trainer = MagicMock()
    trainer.global_rank = 1

    callback.on_train_end(trainer, pl_module)

    model.save_pretrained.assert_not_called()
