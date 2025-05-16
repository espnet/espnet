from pathlib import Path
from unittest import mock

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from espnet3.trainer.callbacks import AverageCheckpointsCallback, get_default_callbacks


@pytest.fixture
def dummy_state_dict():
    return {
        "state_dict": {
            "model.layer.weight": torch.tensor([1.0, 2.0]),
            "model.layer.bias": torch.tensor([0.5]),
            "model.bn.num_batches_tracked": torch.tensor(100, dtype=torch.int64),
        }
    }


def test_average_checkpoints_callback_on_fit_end(tmp_path, dummy_state_dict):
    """
    C001: Ensure AverageCheckpointsCallback correctly averages and saves model.
    """
    ckpt_paths = [tmp_path / f"ckpt_{i}.ckpt" for i in range(2)]

    with mock.patch("torch.load", return_value=dummy_state_dict), mock.patch(
        "torch.save"
    ) as mock_save:

        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[
                mock.Mock(
                    best_k_models={str(p): 0.0 for p in ckpt_paths},
                    monitor="valid/loss",
                )
            ],
        )
        trainer = mock.Mock()
        trainer.is_global_zero = True

        callback.on_fit_end(trainer, pl_module=mock.Mock())

        mock_save.assert_called_once()

        save_path = mock_save.call_args[0][1]
        assert Path(save_path).name.startswith("valid.loss.ave_2best.pth")

        averaged_state = mock_save.call_args[0][0]
        assert torch.allclose(averaged_state["layer.weight"], torch.tensor([1.0, 2.0]))
        assert torch.allclose(averaged_state["layer.bias"], torch.tensor([0.5]))
        assert "bn.num_batches_tracked" in averaged_state


def test_get_default_callbacks_structure():
    """
    C002: Verify the structure and types of callbacks returned.
    """
    callbacks = get_default_callbacks(
        expdir="test_utils/espnet3_dummy/",
        best_model_criterion=[("valid/loss", 2, "min"), ("valid/wer", 2, "min")],
    )

    assert len(callbacks) == 6

    monitor_names = [None, "valid/loss", "valid/wer"]  # None for last checkpoint
    ckpt_callbacks = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint)]
    for cb, expected_monitor in zip(ckpt_callbacks, monitor_names):
        assert cb.monitor == expected_monitor

    has_ave = any(isinstance(cb, AverageCheckpointsCallback) for cb in callbacks)
    assert has_ave
