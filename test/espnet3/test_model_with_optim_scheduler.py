import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from espnet3.trainer.model import LitESPnetModel

# ===============================================================
# Test Case Summary for LitESPnetModel.configure_optimizers
# ===============================================================
#
# Valid Configuration Tests
# | Test Name                           | Description                                                             | # noqa: E501
# |------------------------------------|-------------------------------------------------------------------------| # noqa: E501
# | test_single_optim_and_scheduler    | Validates single optimizer + scheduler configuration                    | # noqa: E501
# | test_multiple_optims_and_schedulers| Validates multiple optimizers and schedulers with param-based mapping   | # noqa: E501
# | test_custom_scheduler_interval     | Ensures scheduler interval is correctly set to "step"                   | # noqa: E501
# | test_reduce_on_plateau_with_config_adam | Tests ReduceLROnPlateau scheduler integration with manual stepping | # noqa: E501
#
# Invalid Configuration Tests
# | Test Name                              | Description                                                             | # noqa: E501
# |---------------------------------------|-------------------------------------------------------------------------| # noqa: E501
# | test_missing_both_optim_and_optims    | Raises ValueError when neither `optim` nor `optims` is defined         | # noqa: E501
# | test_mixed_optim_and_optims           | Detects invalid mix of `optim` and `optims`                             | # noqa: E501
# | test_mixed_scheduler_and_schedulers   | Detects invalid mix of `scheduler` and `schedulers`                     | # noqa: E501
# | test_optims_and_schedulers_length_mismatch | Asserts that number of schedulers matches number of optimizers     | # noqa: E501
# | test_optimizer_missing_params_key     | Ensures `params` key is required for each optimizer config              | # noqa: E501
# | test_optimizer_params_not_matching_model | Asserts at least one parameter is matched for each `params` string   | # noqa: E501
# | test_optimizer_duplicate_params       | Fails if any parameter is assigned to more than one optimizer           | # noqa: E501
# | test_optimizer_missing_coverage       | Detects trainable parameters not assigned to any optimizer              | # noqa: E501
# | test_missing_nested_optim_block       | Ensures each `optims` block contains a nested `optim` definition        | # noqa: E501


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 5)


class ReduceLROnPlateauModel(LitESPnetModel):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.favorite_metric = 1.0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5
        )
        return optimizer

    def optimizer_step(
        self,
        epoch_nb,
        batch_nb,
        optimizer,
    ):
        if batch_nb == 0:
            self.scheduler.step(self.favorite_metric)
            print(
                f"metric: {self.favorite_metric}, best: {self.scheduler.best}, "
                f"num_bad_epochs: {self.scheduler.num_bad_epochs}"
            )
        optimizer.step()
        optimizer.zero_grad()


# ========== VALID CASES ==========


def test_single_optim_and_scheduler():
    config = OmegaConf.create(
        {
            "optim": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 10,
            },
            "dataset": {
                "_target_": "espnet3.data.DataOrganizer",
                "train": [],
                "valid": [],
            },
            "dataloader": {"train": {}, "valid": {}},
            "num_device": 1,
        }
    )
    model = LitESPnetModel(DummyModel(), config)
    out = model.configure_optimizers()
    assert "optimizer" in out
    assert "lr_scheduler" in out


def test_multiple_optims_and_schedulers():
    config = OmegaConf.create(
        {
            "optims": [
                {
                    "optim": {"_target_": "torch.optim.SGD", "lr": 0.01},
                    "params": "linear1",
                },
                {
                    "optim": {"_target_": "torch.optim.Adam", "lr": 0.001},
                    "params": "linear2",
                },
            ],
            "schedulers": [
                {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 10,
                    }
                },
                {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 100,
                    }
                },
            ],
            "dataset": {
                "_target_": "espnet3.data.DataOrganizer",
                "train": [],
                "valid": [],
            },
            "dataloader": {"train": {}, "valid": {}},
            "num_device": 1,
        }
    )
    model = LitESPnetModel(DummyModel(), config)
    out = model.configure_optimizers()
    assert hasattr(out["optimizer"], "optimizers")  # HybridOptim
    assert isinstance(out["lr_scheduler"]["scheduler"], list)


def test_custom_scheduler_interval():
    config = OmegaConf.create(
        {
            "optim": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 5,
            },
            "dataset": {
                "_target_": "espnet3.data.DataOrganizer",
                "train": [],
                "valid": [],
            },
            "dataloader": {"train": {}, "valid": {}},
            "num_device": 1,
        }
    )
    model = LitESPnetModel(DummyModel(), config)
    out = model.configure_optimizers()
    assert out["lr_scheduler"]["interval"] == "step"


def test_reduce_on_plateau_with_config_adam():
    config = OmegaConf.create(
        {
            "optim": {"_target_": "torch.optim.Adam", "lr": 0.01},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "patience": 1,
                "factor": 0.5,
            },
            "dataset": {
                "_target_": "espnet3.data.DataOrganizer",
                "train": [],
                "valid": [],
            },
            "dataloader": {"train": {}, "valid": {}},
            "num_device": 1,
        }
    )

    model = DummyModel()
    lit_model = ReduceLROnPlateauModel(model, config)
    optimizer = lit_model.configure_optimizers()

    # mimic training loop
    for epoch in range(3):
        for batch in range(1):  # we step scheduler at batch 0
            lit_model.favorite_metric = (
                1.0 - 0.1 * epoch
            )  # simulate val_loss decreasing
            lit_model.optimizer_step(
                epoch_nb=epoch,
                batch_nb=batch,
                optimizer=optimizer,
            )


# ========== INVALID CASES ==========


def test_missing_both_optim_and_optims():
    config = OmegaConf.create(
        {
            "dataset": {
                "_target_": "espnet3.data.DataOrganizer",
                "train": [],
                "valid": [],
            },
            "dataloader": {"train": {}, "valid": {}},
            "num_device": 1,
        }
    )
    model = LitESPnetModel(DummyModel(), config)
    with pytest.raises(
        ValueError,
        match="Must specify either `optim` or `optims` and `scheduler` or"
        "`schedulers`",
    ):
        model.configure_optimizers()


def test_mixed_optim_and_optims():
    config = OmegaConf.create(
        {
            "optim": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "optims": [
                {
                    "optim": {"_target_": "torch.optim.SGD", "lr": 0.01},
                    "params": "linear1",
                }
            ],
            "schedulers": [
                {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 10}
            ],
            "dataset": {
                "_target_": "espnet3.data.DataOrganizer",
                "train": [],
                "valid": [],
            },
            "dataloader": {"train": {}, "valid": {}},
            "num_device": 1,
        }
    )
    model = LitESPnetModel(DummyModel(), config)
    with pytest.raises(
        AssertionError, match="Mixture of `optim` and `optims` is not allowed"
    ):
        model.configure_optimizers()


def test_mixed_scheduler_and_schedulers():
    config = OmegaConf.create(
        {
            "optim": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 10,
            },
            "schedulers": [
                {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 10,
                    }
                }
            ],
            "dataset": {
                "_target_": "espnet3.data.DataOrganizer",
                "train": [],
                "valid": [],
            },
            "dataloader": {"train": {}, "valid": {}},
            "num_device": 1,
        }
    )
    model = LitESPnetModel(DummyModel(), config)
    with pytest.raises(
        AssertionError, match="Mixture of `scheduler` and `schedulers` is not allowed"
    ):
        model.configure_optimizers()


def test_optims_and_schedulers_length_mismatch():
    config = OmegaConf.create(
        {
            "optims": [
                {
                    "optim": {"_target_": "torch.optim.Adam", "lr": 0.001},
                    "params": "linear1",
                },
                {
                    "optim": {"_target_": "torch.optim.SGD", "lr": 0.01},
                    "params": "linear2",
                },
            ],
            "schedulers": [
                {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 10,
                    }
                }
            ],
            "dataset": {
                "_target_": "espnet3.data.DataOrganizer",
                "train": [],
                "valid": [],
            },
            "dataloader": {"train": {}, "valid": {}},
            "num_device": 1,
        }
    )
    model = LitESPnetModel(DummyModel(), config)
    with pytest.raises(
        AssertionError, match="The number of optimizers and schedulers must be equal"
    ):
        model.configure_optimizers()


def test_optimizer_missing_params_key():
    config = OmegaConf.create(
        {
            "optims": [
                {"optim": {"_target_": "torch.optim.SGD", "lr": 0.01}}
            ],  # Missing "params"
            "schedulers": [
                {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 10,
                    }
                }
            ],
            "dataset": {
                "_target_": "espnet3.data.DataOrganizer",
                "train": [],
                "valid": [],
            },
            "dataloader": {"train": {}, "valid": {}},
            "num_device": 1,
        }
    )
    model = LitESPnetModel(DummyModel(), config)
    with pytest.raises(AssertionError, match="missing 'params' in optim config"):
        model.configure_optimizers()


def test_optimizer_params_not_matching_model():
    config = OmegaConf.create(
        {
            "optims": [
                {
                    "optim": {"_target_": "torch.optim.SGD", "lr": 0.01},
                    "params": "does_not_exist",
                }
            ],
            "schedulers": [
                {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 10,
                    }
                }
            ],
            "dataset": {
                "_target_": "espnet3.data.DataOrganizer",
                "train": [],
                "valid": [],
            },
            "dataloader": {"train": {}, "valid": {}},
            "num_device": 1,
        }
    )
    model = LitESPnetModel(DummyModel(), config)
    with pytest.raises(AssertionError, match="No trainable parameters found for"):
        model.configure_optimizers()


def test_optimizer_duplicate_params():
    config = OmegaConf.create(
        {
            "optims": [
                {
                    "optim": {"_target_": "torch.optim.Adam", "lr": 0.001},
                    "params": "linear",  # matches both linear1 and linear2
                },
                {
                    "optim": {"_target_": "torch.optim.SGD", "lr": 0.01},
                    "params": "linear",  # same
                },
            ],
            "schedulers": [
                {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 10,
                    }
                },
                {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 10,
                    }
                },
            ],
            "dataset": {
                "_target_": "espnet3.data.DataOrganizer",
                "train": [],
                "valid": [],
            },
            "dataloader": {"train": {}, "valid": {}},
            "num_device": 1,
        }
    )
    model = LitESPnetModel(DummyModel(), config)
    with pytest.raises(
        AssertionError,
        match="Parameter model.linear1.weight is assigned to multiple optimizers",
    ):
        model.configure_optimizers()


def test_optimizer_missing_coverage():
    class PartialModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 10)
            self.linear2 = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear1(x)

    config = OmegaConf.create(
        {
            "optims": [
                {
                    "optim": {"_target_": "torch.optim.Adam", "lr": 0.001},
                    "params": "linear1",
                }
            ],
            "schedulers": [
                {
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 10,
                    }
                },
            ],
            "dataset": {
                "_target_": "espnet3.data.DataOrganizer",
                "train": [],
                "valid": [],
            },
            "dataloader": {"train": {}, "valid": {}},
            "num_device": 1,
        }
    )
    model = LitESPnetModel(PartialModel(), config)
    with pytest.raises(AssertionError) as excinfo:
        model.configure_optimizers()
    assert "model.linear2.bias" in str(excinfo.value)
    assert "model.linear2.weight" in str(excinfo.value)
