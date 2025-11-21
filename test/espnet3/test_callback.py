from pathlib import Path
from unittest import mock

import pytest
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf

from espnet3.trainer.callbacks import AverageCheckpointsCallback, get_default_callbacks

# ===============================================================
# Test Case Summary for AverageCheckpointsCallback
# ===============================================================
#
# Normal Cases
# | Test Name                                      | Description                       |
# |-----------------------------------------------|------------------------------------|
# | test_average_checkpoints_callback_on_validation_end  | Verifies that checkpoint    |
# |                        | averaging and saving works correctly with dummy weights.  |
# | test_get_default_callbacks_structure          | Checks structure and types of      |
# |                                   | callbacks returned by get_default_callbacks(). |
# | test_average_checkpoints_with_multiple_metrics| Confirms correct averaging for     |
# |                |  multiple ModelCheckpoint instances with different monitor names. |
# | test_output_filename_format                 | Ensures output filename is formatted |
# |                                         | using monitor name and checkpoint count. |
# | test_duplicate_learning_rate_monitor_from_config | Confirms that if                |
# | |LearningRateMonitor is defined both by default and in the config, duplicates occur|
# | | (no deduplication or warning yet).                                       |
#
# Edge/Error Cases
# | Test Name                                      | Description                       |
# |-----------------------------------------------|------------------------------------|
# | test_average_checkpoint_on_non_global_zero    | Ensures callback is skipped when   |
# |                       | trainer.is_global_zero is False (e.g., non-main DDP rank). |
# | test_average_checkpoint_with_inconsistent_keys| Raises KeyError if state_dict keys |
# |                                               | differ across checkpoints. |
# | test_average_checkpoint_with_int_and_float_mix| Confirms floats are averaged and   |
# |                          | ints are accumulated properly during checkpoint merging.|


@pytest.fixture
def dummy_state_dict():
    return {
        "state_dict": {
            "model.layer.weight": torch.tensor([1.0, 2.0]),
            "model.layer.bias": torch.tensor([0.5]),
            "model.bn.num_batches_tracked": torch.tensor(100, dtype=torch.int64),
        }
    }


def test_average_checkpoints_callback_on_validation_end(tmp_path, dummy_state_dict):
    """Test average checkpoints.

    Ensure AverageCheckpointsCallback correctly averages and saves model.
    """
    ckpt_paths = [tmp_path / f"ckpt_{i}.ckpt" for i in range(2)]

    with (
        mock.patch("torch.load", return_value=dummy_state_dict),
        mock.patch("torch.save") as mock_save,
    ):

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

        callback.on_validation_end(trainer, pl_module=mock.Mock())

        mock_save.assert_called_once()

        save_path = mock_save.call_args[0][1]
        assert Path(save_path).name.startswith("valid.loss.ave_2best.pth")

        averaged_state = mock_save.call_args[0][0]
        assert torch.allclose(averaged_state["layer.weight"], torch.tensor([1.0, 2.0]))
        assert torch.allclose(averaged_state["layer.bias"], torch.tensor([0.5]))
        assert "bn.num_batches_tracked" in averaged_state


def test_get_default_callbacks_structure():
    """Test Get default callbacks.

    Verify the structure and types of callbacks returned.
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


def test_average_checkpoints_with_multiple_metrics(tmp_path, dummy_state_dict):
    """Test averaging for multiple ModelCheckpoints with different monitor names."""
    ckpt_paths_1 = [tmp_path / f"ckpt_loss_{i}.ckpt" for i in range(2)]
    ckpt_paths_2 = [tmp_path / f"ckpt_acc_{i}.ckpt" for i in range(2)]

    with (
        mock.patch("torch.load", return_value=dummy_state_dict),
        mock.patch("torch.save") as mock_save,
    ):
        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[
                mock.Mock(
                    best_k_models={str(p): 0.0 for p in ckpt_paths_1},
                    monitor="valid/loss",
                ),
                mock.Mock(
                    best_k_models={str(p): 0.0 for p in ckpt_paths_2},
                    monitor="valid/acc",
                ),
            ],
        )
        trainer = mock.Mock(is_global_zero=True)
        callback.on_validation_end(trainer, pl_module=mock.Mock())

        assert mock_save.call_count == 2
        filenames = [Path(call.args[1]).name for call in mock_save.call_args_list]
        assert "valid.loss.ave_2best.pth" in filenames
        assert "valid.acc.ave_2best.pth" in filenames


def test_output_filename_format(tmp_path, dummy_state_dict):
    """Ensure output filename is formatted properly."""
    ckpt_paths = [tmp_path / f"ckpt_{i}.ckpt" for i in range(3)]

    with (
        mock.patch("torch.load", return_value=dummy_state_dict),
        mock.patch("torch.save") as mock_save,
    ):
        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[
                mock.Mock(
                    best_k_models={str(p): 0.0 for p in ckpt_paths},
                    monitor="some/metric",
                )
            ],
        )
        trainer = mock.Mock(is_global_zero=True)
        callback.on_validation_end(trainer, pl_module=mock.Mock())

        filename = Path(mock_save.call_args[0][1]).name
        assert filename == "some.metric.ave_3best.pth"


def test_average_checkpoint_on_non_global_zero(tmp_path, dummy_state_dict):
    """Ensure averaging does nothing when not global rank 0."""
    with (
        mock.patch("torch.load", return_value=dummy_state_dict),
        mock.patch("torch.save") as mock_save,
    ):
        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[
                mock.Mock(best_k_models={"dummy.ckpt": 0.0}, monitor="valid/loss")
            ],
        )
        trainer = mock.Mock(is_global_zero=False)
        callback.on_validation_end(trainer, pl_module=mock.Mock())

        mock_save.assert_not_called()


def test_average_checkpoint_with_inconsistent_keys(tmp_path):
    """Raise error when checkpoints have inconsistent keys."""
    ckpt_path1 = tmp_path / "ckpt_1.ckpt"
    ckpt_path2 = tmp_path / "ckpt_2.ckpt"

    inconsistent_state_dicts = [
        {"state_dict": {"model.layer.weight": torch.tensor([1.0])}},  # 1 key
        {
            "state_dict": {
                "model.layer.weight": torch.tensor([1.0]),
                "model.layer.bias": torch.tensor([0.5]),
            }
        },
    ]

    def load_side_effect(path, *args, **kwargs):
        return inconsistent_state_dicts.pop(0)

    with (
        mock.patch("torch.load", side_effect=load_side_effect),
        pytest.raises(KeyError),
    ):
        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[
                mock.Mock(
                    best_k_models={str(ckpt_path1): 0.0, str(ckpt_path2): 0.0},
                    monitor="valid/loss",
                )
            ],
        )
        trainer = mock.Mock(is_global_zero=True)
        callback.on_validation_end(trainer, pl_module=mock.Mock())


def test_average_checkpoint_with_int_and_float_mix(tmp_path):
    """Ensure float params are averaged, int params are accumulated."""
    ckpt_path1 = tmp_path / "ckpt_1.ckpt"
    ckpt_path2 = tmp_path / "ckpt_2.ckpt"

    mock_state_dicts = [
        {
            "state_dict": {
                "model.weight": torch.tensor([2.0, 4.0]),
                "model.counter": torch.tensor(10, dtype=torch.int64),
            }
        },
        {
            "state_dict": {
                "model.weight": torch.tensor([6.0, 2.0]),
                "model.counter": torch.tensor(30, dtype=torch.int64),
            }
        },
    ]

    def load_side_effect(path, *args, **kwargs):
        return mock_state_dicts.pop(0)

    with (
        mock.patch("torch.load", side_effect=load_side_effect),
        mock.patch("torch.save") as mock_save,
    ):
        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[
                mock.Mock(
                    best_k_models={str(ckpt_path1): 0.0, str(ckpt_path2): 0.0},
                    monitor="valid/loss",
                )
            ],
        )
        trainer = mock.Mock(is_global_zero=True)
        callback.on_validation_end(trainer, pl_module=mock.Mock())

        saved = mock_save.call_args[0][0]
        # Float averaged
        assert torch.allclose(saved["weight"], torch.tensor([4.0, 3.0]))
        # Int not averaged
        assert saved["counter"] == 40


def test_average_checkpoint_with_no_checkpoints(tmp_path):
    """Ensure averaging does nothing when there are no checkpoints."""
    with mock.patch("torch.save") as mock_save:
        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[mock.Mock(best_k_models={}, monitor="valid/loss")],
        )
        trainer = mock.Mock(is_global_zero=True)
        # This should not raise an exception
        callback.on_validation_end(trainer, pl_module=mock.Mock())

        mock_save.assert_not_called()


def test_duplicate_learning_rate_monitor_from_config():
    """Test duplicate LearningRateMonitor creation.

    Verify that when a LearningRateMonitor is provided both by default and in the user
    configuration, two separate instances are created. The current behavior does not
    emit a warning or perform any deduplication of these callbacks.
    """
    # First, get the default callbacks (contains exactly one LearningRateMonitor)
    callbacks = get_default_callbacks(
        expdir="test_utils/espnet3_dummy/",
        best_model_criterion=[("valid/loss", 2, "min")],
    )
    # Ensure only one LearningRateMonitor is included by default
    assert sum(isinstance(cb, LearningRateMonitor) for cb in callbacks) == 1

    # Simulate specifying LearningRateMonitor again via config (Hydra-style)
    cfg = OmegaConf.create(
        {"callbacks": [{"_target_": "lightning.pytorch.callbacks.LearningRateMonitor"}]}
    )
    # Append the instantiated callback to mimic trainer logic
    for cb_conf in cfg.callbacks:
        callbacks.append(instantiate(cb_conf))

    # Now we should have duplicates (2 LearningRateMonitor instances)
    # because no deduplication or warning is implemented yet
    assert sum(isinstance(cb, LearningRateMonitor) for cb in callbacks) == 2

    # AverageCheckpointsCallback should still be exactly one (unaffected by duplicates)
    assert sum(isinstance(cb, AverageCheckpointsCallback) for cb in callbacks) == 1
