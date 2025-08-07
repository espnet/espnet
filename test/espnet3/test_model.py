import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from espnet3.trainer.model import LitESPnetModel

# ===============================================================
# Test Case Summary for LitESPnetModel
# ===============================================================
#
# Normal Cases
# | Test Name                          | Description                                                              | # noqa: E501
# |-----------------------------------|--------------------------------------------------------------------------| # noqa: E501
# | test_training_step_runs           | Verifies that training_step executes and returns a loss                  | # noqa: E501
# | test_validation_step_runs         | Verifies that validation_step runs and returns a loss                    | # noqa: E501
# | test_is_espnet_sampler_flag       | Checks that is_espnet_sampler is set correctly based on iter_factory     | # noqa: E501
# | test_state_dict_and_load          | Ensures state_dict and load_state_dict are correctly forwarded           | # noqa: E501
# | test_use_espnet_collator_flag_false | Checks dataset.use_espnet_collator=False for non-CommonCollateFn        | # noqa: E501
#
# Error / Edge Cases
# | Test Name                          | Description                                                              | # noqa: E501
# |-----------------------------------|--------------------------------------------------------------------------| # noqa: E501
# | test_nan_loss_skip                | Ensures batch is skipped when loss is NaN/Inf                            | # noqa: E501
# | test_dataloader_mismatch_raises  | Asserts error if train/valid dataloader types mismatch                   | # noqa: E501
# | test_mixed_optim_scheduler_raises| Asserts error if both optim and optims/schedulers are defined            | # noqa: E501
# | test_missing_optimizer_and_scheduler_raises | Raises ValueError when neither optimizer nor scheduler is specified     | # noqa: E501
#

# ---------------------- Dummy Components ----------------------


class DummyDataset:
    def __init__(self, path=None):
        self.data = [
            {
                "id": "utt_a",
                "audio": np.random.rand(16000).astype(np.float32),
                "text": np.array([1, 2, 3]),
            },
            {
                "id": "utt_b",
                "audio": np.random.rand(32000).astype(np.float32),
                "text": np.array([1, 2, 3, 4]),
            },
            {
                "id": "utt_c",
                "audio": np.random.rand(48000).astype(np.float32),
                "text": np.array([1, 2, 3, 4, 5]),
            },
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "audio": self.data[idx]["audio"],
            "text": self.data[idx]["text"],
        }


@pytest.fixture
def dummy_model():
    model = torch.nn.Linear(1, 1)
    model.forward = lambda **kwargs: (
        torch.tensor(0.123),
        {"loss": torch.tensor(0.123)},
        torch.tensor(1.0),
    )
    return model


@pytest.fixture
def dummy_dataset_config():
    return OmegaConf.create(
        {
            "_target_": "espnet3.data.DataOrganizer",
            "train": [
                {
                    "name": "dummy_train",
                    "dataset": {"_target_": __name__ + ".DummyDataset"},
                }
            ],
            "valid": [
                {
                    "name": "dummy_valid",
                    "dataset": {"_target_": __name__ + ".DummyDataset"},
                }
            ],
        }
    )


def make_standard_dataloader_config():
    dataloader_config = dict(
        batch_size=2,
        shuffle=False,
        drop_last=False,
        iter_factory=None,
    )
    config = dict(
        collate_fn={
            "_target_": "espnet2.train.collate_fn.CommonCollateFn",
            "int_pad_value": -1,
        },
        train=dataloader_config,
        valid=dataloader_config,
    )
    return OmegaConf.create(config)


# ---------------------- Normal Cases ----------------------


def test_training_step_runs(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "expdir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = LitESPnetModel(dummy_model, config)
    batch = next(iter(model.train_dataloader()))
    out = model.training_step(batch, 0)
    assert np.allclose(out.item(), 0.123)


def test_validation_step_runs(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "expdir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = LitESPnetModel(dummy_model, config)
    batch = next(iter(model.val_dataloader()))
    out = model.validation_step(batch, 0)
    assert torch.is_tensor(out)


def test_is_espnet_sampler_flag(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "expdir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": {
                "train": {"iter_factory": None},
                "valid": {"iter_factory": None},
            },
            "num_device": 1,
        }
    )
    model = LitESPnetModel(dummy_model, config)
    assert model.is_espnet_sampler is False


def test_state_dict_and_load(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "expdir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = LitESPnetModel(dummy_model, config)
    state = model.state_dict()
    model.load_state_dict(state)


class CustomCollate:
    def __init__(self):
        pass

    def __call__(batch):
        return {"custom": True}


def test_use_espnet_collator_flag_false(tmp_path, dummy_model, dummy_dataset_config):

    config = OmegaConf.create(
        {
            "expdir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": {
                "collate_fn": {"_target_": "test.espnet3.test_model.CustomCollate"},
                "train": {"iter_factory": None},
                "valid": {"iter_factory": None},
            },
            "num_device": 1,
        }
    )
    model = LitESPnetModel(dummy_model, config)
    _ = model.train_dataloader()
    _ = model.val_dataloader()
    assert model.train_dataset.use_espnet_collator is False
    assert model.valid_dataset.use_espnet_collator is False


# ---------------------- Error Cases ----------------------


def test_nan_loss_skip(tmp_path, dummy_dataset_config):
    def nan_model(**kwargs):
        return (
            torch.tensor(float("nan")),
            {"loss": torch.tensor(float("nan"))},
            torch.tensor(1.0),
        )

    dummy_model = torch.nn.Linear(1, 1)
    dummy_model.forward = nan_model

    config = OmegaConf.create(
        {
            "expdir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = LitESPnetModel(dummy_model, config)
    batch = next(iter(model.train_dataloader()))
    out = model.training_step(batch, 0)
    assert out is None


def test_dataloader_mismatch_raises(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "expdir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": {
                "train": {"iter_factory": None},
                "valid": {"iter_factory": {"_target_": "dummy"}},
            },
            "num_device": 1,
        }
    )
    with pytest.raises(
        AssertionError, match="Train and valid should have the same type of dataloader"
    ):
        _ = LitESPnetModel(dummy_model, config)


def test_mixed_optim_scheduler_raises(tmp_path, dummy_model, dummy_dataset_config):
    config = OmegaConf.create(
        {
            "expdir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "optim": {"_target_": "torch.optim.Adam"},
            "optims": [{"_target_": "torch.optim.SGD"}],
            "scheduler": {"_target_": "torch.optim.lr_scheduler.StepLR"},
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
        }
    )
    model = LitESPnetModel(dummy_model, config)
    with pytest.raises(
        AssertionError, match="Mixture of `optim` and `optims` is not allowed."
    ):
        model.configure_optimizers()


def test_missing_optimizer_and_scheduler_raises(
    tmp_path, dummy_model, dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "expdir": str(tmp_path / "exp"),
            "dataset": dummy_dataset_config,
            "dataloader": make_standard_dataloader_config(),
            "num_device": 1,
            # intentionally omit both `optim`, `optims`, and `scheduler`, `schedulers`
        }
    )
    model = LitESPnetModel(dummy_model, config)
    with pytest.raises(
        ValueError, match="Must specify either `optim` or `optims` and `scheduler` or"
    ):
        model.configure_optimizers()
