import pytest
from unittest.mock import MagicMock

from espnet3.components.callbacks.hf_callbacks import HFCheckpointSaveCallback
from espnet3.components.modeling.hf_models import AbsHFTrainingWrapper

class TestHFWrapper(AbsHFTrainingWrapper):
    """Wrapper for testing functionality without downloading models from HF."""
    def collect_feats(self, **batch):
        pass


def test_hf_checkpoint_save_on_train_end(tmp_path):
    """Test saving Hugging Face model checkpoints."""
    callback = HFCheckpointSaveCallback(str(tmp_path))

    # __init__() would download a model and processor from HF.
    # Since we don't need those here, we call __new__(), which
    # instantiates an object without calling __init__(), saving
    # us a costly operation.
    model = TestHFWrapper.__new__(TestHFWrapper)
    model.save_pretrained = MagicMock()

    pl_module = MagicMock()
    pl_module.model = model

    trainer = MagicMock()
    trainer.global_rank = 0

    callback.on_train_end(trainer, pl_module)

    model.save_pretrained.assert_called_once_with(str(tmp_path))


def test_hf_checkpoint_save_invalid_model_class(tmp_path):
    """Test that callback only accepts HF training wrappers."""
    callback = HFCheckpointSaveCallback(str(tmp_path))

    pl_module = MagicMock()
    pl_module.model = MagicMock()

    trainer = MagicMock()
    trainer.global_rank = 0

    with pytest.raises(AttributeError):
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

