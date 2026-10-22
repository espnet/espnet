from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from omegaconf import OmegaConf

from espnet3.components.callbacks.default_callbacks import (
    AverageCheckpointsCallback,
    MetricsLogger,
    _metric_to_float,
    get_default_callbacks,
)
from espnet3.components.data import data_organizer as data_organizer_module
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer

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


def _mock_pl_module(*keys):
    """Build a pl_module mock whose `state_dict()` exposes the given keys.

    `AverageCheckpointsCallback` uses `pl_module.state_dict().keys()` as the
    reference key set to decide whether a `model.` prefix must be stripped
    from the loaded checkpoints.
    """
    module = mock.Mock()
    module.state_dict.return_value = {key: None for key in keys}
    return module


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

        pl_module = _mock_pl_module(
            "layer.weight", "layer.bias", "bn.num_batches_tracked"
        )
        callback.on_validation_end(trainer, pl_module=pl_module)

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
        exp_dir="test_utils/espnet3_dummy/",
        best_model_criterion=[("valid/loss", 2, "min"), ("valid/wer", 2, "min")],
    )

    assert len(callbacks) == 7

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
        pl_module = _mock_pl_module(
            "layer.weight", "layer.bias", "bn.num_batches_tracked"
        )
        callback.on_validation_end(trainer, pl_module=pl_module)

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
        pl_module = _mock_pl_module(
            "layer.weight", "layer.bias", "bn.num_batches_tracked"
        )
        callback.on_validation_end(trainer, pl_module=pl_module)

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
        pl_module = _mock_pl_module("layer.weight")
        callback.on_validation_end(trainer, pl_module=pl_module)


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
        pl_module = _mock_pl_module("weight", "counter")
        callback.on_validation_end(trainer, pl_module=pl_module)

        saved = mock_save.call_args[0][0]
        # Float averaged
        assert torch.allclose(saved["weight"], torch.tensor([4.0, 3.0]))
        # Int not averaged
        assert saved["counter"] == 40


def test_average_checkpoint_without_model_prefix(tmp_path):
    """Average checkpoints whose keys have no `model.` prefix.

    Models built via an espnet2 Task (e.g. `ASRTask.build_model`) expose their
    own submodules (`frontend`, `encoder`, `decoder`, ...) directly, so
    `ESPnetLightningModule.state_dict()` returns keys without a `model.`
    prefix. Averaging must not silently drop these parameters.
    """
    ckpt_path1 = tmp_path / "ckpt_1.ckpt"
    ckpt_path2 = tmp_path / "ckpt_2.ckpt"

    mock_state_dicts = [
        {
            "state_dict": {
                "encoder.weight": torch.tensor([2.0, 4.0]),
                "decoder.bias": torch.tensor([1.0]),
            }
        },
        {
            "state_dict": {
                "encoder.weight": torch.tensor([6.0, 2.0]),
                "decoder.bias": torch.tensor([3.0]),
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
        pl_module = _mock_pl_module("encoder.weight", "decoder.bias")
        callback.on_validation_end(trainer, pl_module=pl_module)

        saved = mock_save.call_args[0][0]
        assert torch.allclose(saved["encoder.weight"], torch.tensor([4.0, 3.0]))
        assert torch.allclose(saved["decoder.bias"], torch.tensor([2.0]))


def test_average_checkpoint_keys_mismatch_current_model(tmp_path, dummy_state_dict):
    """Raise a KeyError when checkpoint keys never match the live model.

    Neither the raw checkpoint keys nor the `model.`-stripped keys match
    `pl_module.state_dict()`, which signals a genuine mismatch (e.g. loading
    checkpoints saved for a different model architecture) rather than the
    ordinary `model.`-prefix ambiguity.
    """
    ckpt_paths = [tmp_path / f"ckpt_{i}.ckpt" for i in range(2)]

    with (
        mock.patch("torch.load", return_value=dummy_state_dict),
        mock.patch("torch.save"),
        pytest.raises(KeyError),
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
        trainer = mock.Mock(is_global_zero=True)
        pl_module = _mock_pl_module("totally.unrelated.param")
        callback.on_validation_end(trainer, pl_module=pl_module)


def test_average_checkpoint_with_no_checkpoints(tmp_path):
    """Ensure averaging does nothing when there are no checkpoints."""
    with mock.patch("torch.save") as mock_save:
        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[mock.Mock(best_k_models={}, monitor="valid/loss")],
        )
        trainer = mock.Mock(is_global_zero=True)
        # This should not raise an exception
        callback.on_validation_end(trainer, pl_module=_mock_pl_module())

        mock_save.assert_not_called()


def test_duplicate_learning_rate_monitor_from_config():
    """Test duplicate LearningRateMonitor creation.

    Verify that when a LearningRateMonitor is provided both by default and in the user
    configuration, two separate instances are created. The current behavior does not
    emit a warning or perform any deduplication of these callbacks.
    """
    # First, get the default callbacks (contains exactly one LearningRateMonitor)
    callbacks = get_default_callbacks(
        exp_dir="test_utils/espnet3_dummy/",
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


def test_metric_to_float_rejects_non_scalar_tensor():
    with pytest.raises(AssertionError, match="supports only scalar metric values"):
        _metric_to_float(torch.tensor([1.0, 2.0]))


def test_metric_to_float_rejects_unsupported_type():
    with pytest.raises(
        AssertionError, match="does not support metric values of type dict"
    ):
        _metric_to_float({"loss": 1.0})


# ---------------------------------------------------------------
# trainer-callbacks#07: end-to-end coverage with a real Lightning run
# ---------------------------------------------------------------


class _TinyRegressionDataset:
    """4 deterministic (x, y=2x) points; both x and y are model kwargs.

    Values are numpy arrays (not torch tensors) because the default
    collate_fn is CommonCollateFn, which inspects ``.dtype.kind`` -- a numpy
    attribute that a bare ``torch.Tensor`` does not have.
    """

    def __init__(self, path=None):
        self.data = [
            {
                "x": np.array([float(i)], dtype=np.float32),
                "y": np.array([2.0 * i], dtype=np.float32),
            }
            for i in range(1, 5)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class _TinyRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, y, **kwargs):
        pred = self.linear(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        return loss, {"loss": loss.detach()}, None


@pytest.fixture
def _patch_tiny_dataset(monkeypatch):
    monkeypatch.setattr(
        data_organizer_module,
        "instantiate_dataset_reference",
        lambda config, recipe_dir=None: _TinyRegressionDataset(),
    )


@pytest.mark.execution_timeout(60)
def test_default_callbacks_e2e_produces_real_best_and_averaged_checkpoints(
    tmp_path, _patch_tiny_dataset, monkeypatch
):
    """Drive 3 real training epochs through get_default_callbacks() and CSVLogger.

    Regression coverage for trainer-callbacks#07: every existing test in this
    file replaces ModelCheckpoint with mocks and calls
    AverageCheckpointsCallback.on_validation_end() directly with
    torch.load/save mocked out, so a real Lightning run was never exercised.
    This test trains a tiny real model for 3 epochs and asserts:
    (1) the top-2 best checkpoints and the averaged ``.pth`` both exist,
    (2) the averaged weights equal the real mean of whichever 2 checkpoints
        were actually used for the last averaging pass (no mocking), and
    (3) get_default_callbacks()'s returned list matches its documented order
        contract.

    Discovered while writing this test: Lightning's CallbackConnector moves
    every ``ModelCheckpoint`` to the *end* of the runtime callback list
    regardless of the order passed to ``Trainer(callbacks=...)``, so
    ``AverageCheckpointsCallback.on_validation_end`` always runs with the
    *previous* epoch's ``best_k_models`` (ModelCheckpoint for the current
    epoch has not saved yet). The averaged ``.pth`` is therefore always one
    epoch stale relative to the freshest checkpoint. This does not corrupt
    the average (it is still a real mean of 2 real top-k checkpoints), but it
    means "the current epoch's checkpoint" and "what got averaged" can
    diverge by one epoch -- worth being aware of when comparing average vs.
    best-checkpoint metrics. Assertion (3) below therefore checks the
    *construction-time* order contract from ``get_default_callbacks()``
    itself, not runtime ``trainer.callbacks`` order, since the latter is
    reshuffled by Lightning, not by ESPnet3.

    EMA-enabled verification is out of scope here (not wired into
    get_default_callbacks by default); see the finding's outlook.
    """
    exp_dir = tmp_path / "exp"
    model_config = OmegaConf.create(
        {
            "exp_dir": str(exp_dir),
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 10,
            },
            "dataset": {
                "_target_": "espnet3.components.data.data_organizer.DataOrganizer",
                "train": [{"name": "train_dummy", "data_src": "dummy/regression"}],
                "valid": [{"name": "valid_dummy", "data_src": "dummy/regression"}],
            },
            "dataloader": {
                "train": {"batch_size": 4, "shuffle": False, "iter_factory": None},
                "valid": {"batch_size": 4, "shuffle": False, "iter_factory": None},
            },
            "num_device": 1,
        }
    )
    module = ESPnetLightningModule(_TinyRegressionModel(), model_config)

    trainer_config = OmegaConf.create(
        {
            "accelerator": "cpu",
            "devices": 1,
            "num_nodes": 1,
            "max_epochs": 3,
            "num_sanity_val_steps": 0,
            "log_every_n_steps": 1,
            "logger": {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": str(tmp_path / "csv"),
                "name": "e2e",
            },
        }
    )
    wrapper = ESPnet3LightningTrainer(
        model=module,
        exp_dir=str(exp_dir),
        config=trainer_config,
        best_model_criterion=OmegaConf.create([["valid/loss", 2, "min"]]),
    )

    # Record exactly which checkpoints were on hand each time
    # AverageCheckpointsCallback actually ran, so assertion (2) below can
    # compare against the real inputs to the *last* averaging pass rather
    # than assuming it matches the final (post-training) best_k_models.
    # Checkpoints get evicted by later top-k pruning, so load their tensors
    # now (while the files still exist) rather than storing paths to load
    # after training completes.
    seen_checkpoint_states = []
    original_on_validation_end = AverageCheckpointsCallback.on_validation_end

    def _recording_on_validation_end(self, trainer, pl_module):
        for ckpt_callback in self.best_ckpt_callbacks:
            seen_checkpoint_states.append(
                [
                    torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
                    for p in ckpt_callback.best_k_models
                ]
            )
        return original_on_validation_end(self, trainer, pl_module)

    monkeypatch.setattr(
        AverageCheckpointsCallback, "on_validation_end", _recording_on_validation_end
    )

    wrapper.fit()

    callbacks = wrapper.trainer.callbacks
    ckpt_callbacks = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint)]
    best_ckpt_callback = next(cb for cb in ckpt_callbacks if cb.monitor == "valid/loss")

    # (1) files exist
    assert len(best_ckpt_callback.best_k_models) == 2
    for ckpt_path in best_ckpt_callback.best_k_models:
        assert Path(ckpt_path).is_file()
    ave_path = exp_dir / "valid.loss.ave_2best.pth"
    assert ave_path.is_file()

    # (2) the averaged weights are the real mean of whichever 2 checkpoints
    # were actually on hand for the last averaging pass (see docstring).
    real_states = seen_checkpoint_states[-1]
    assert len(real_states) == 2
    # ESPnetLightningModule.state_dict() delegates directly to the wrapped
    # model's own state_dict(), so checkpoint keys are unprefixed ("linear.*",
    # not "model.linear.*").
    expected_weight = sum(s["linear.weight"] for s in real_states) / len(real_states)
    expected_bias = sum(s["linear.bias"] for s in real_states) / len(real_states)
    averaged = torch.load(ave_path, map_location="cpu", weights_only=False)
    assert torch.allclose(averaged["linear.weight"], expected_weight)
    assert torch.allclose(averaged["linear.bias"], expected_bias)

    # (3) get_default_callbacks()'s own returned list follows its documented
    # order contract (last-ckpt, best-ckpt(s), average, LR monitor,
    # MetricsLogger, progress bar). This is the ESPnet3-owned contract;
    # Lightning's runtime reordering of ModelCheckpoint (see docstring above)
    # is Lightning's behavior, not ESPnet3's, so it is not asserted here.
    constructed = get_default_callbacks(
        exp_dir=str(exp_dir), best_model_criterion=[("valid/loss", 2, "min")]
    )
    kinds = [type(cb) for cb in constructed]
    assert kinds.index(ModelCheckpoint) < kinds.index(AverageCheckpointsCallback)
    assert kinds.index(AverageCheckpointsCallback) < kinds.index(LearningRateMonitor)
    assert kinds.index(LearningRateMonitor) < kinds.index(MetricsLogger)
    assert kinds.index(MetricsLogger) < kinds.index(TQDMProgressBar)
