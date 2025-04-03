import os
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from espnet3.trainer import LitESPnetModel


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return (str(index), {"x": np.array([index])})

    def __len__(self):
        return 10


@pytest.fixture
def dummy_model():
    model = torch.nn.Linear(1, 1)
    return model


@pytest.mark.parametrize(
    "config_dict",
    [
        {
            # 標準的な構成
            "expdir": "exp",
            "dataloader": {
                "collate_fn": {
                    "_target_": "espnet2.train.collate_fn.CommonCollateFn",
                    "int_pad_value": -1,
                },
                "train": {"shuffle": True, "batch_size": 2, "num_workers": 0},
                "valid": {"shuffle": False, "batch_size": 1, "num_workers": 0},
            },
        },
        {
            # サンプラーを使うパターン（ランダムサンプラー）
            "expdir": "exp",
            "dataloader": {
                "train": {
                    "batch_size": 3,
                    "num_workers": 0,
                    "sampler": {"_target_": "torch.utils.data.RandomSampler"},
                }
            },
        },
    ],
)
def test_dataloader_configs(config_dict, dummy_model):
    config = OmegaConf.create(config_dict)
    dataset = DummyDataset()
    model = LitESPnetModel(dummy_model, config, dataset, dataset)

    train_dl = model.train_dataloader()
    batch = next(iter(train_dl))

    # batch_size確認（collate_fnが返すtupleの長さ）
    assert isinstance(batch, (list, tuple))
    assert len(batch) == 2

    # collate_fnが指定通り
    if "collate_fn" in config.dataloader:
        assert hasattr(train_dl.collate_fn, "__call__")
        assert hasattr(train_dl.collate_fn, "int_pad_value")
        assert train_dl.collate_fn.int_pad_value == -1

    # samplerが正しく構築されているか
    if "sampler" in config.dataloader.get("train", {}):
        from torch.utils.data import RandomSampler

        assert isinstance(train_dl.sampler, RandomSampler)


@pytest.fixture
def config(tmp_path):
    return OmegaConf.create(
        {
            "expdir": str(tmp_path / "exp"),
            "optim": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
            },
            "dataloader": {
                "collate_fn": {
                    "_target_": "espnet2.train.collate_fn.CommonCollateFn",
                    "int_pad_value": -1,
                },
                "train": {
                    "batch_size": 80,
                    "num_workers": 0,  # pytest内では0の方が安全
                    "shuffle": True,
                },
                "valid": {
                    "batch_size": 4,
                    "num_workers": 0,
                    "shuffle": False,
                },
            },
        }
    )


@pytest.fixture
def dummy_model():
    # forward が (loss, stats_dict, weight) を返すようにモック
    model = torch.nn.Linear(1, 1)
    model.forward = lambda **kwargs: (
        torch.tensor(0.123),
        {"loss": torch.tensor(0.123)},
        torch.tensor(1.0),
    )
    return model


@pytest.fixture
def dummy_dataset():
    return DummyDataset()


def test_train_dataloader_structure(config, dummy_model, dummy_dataset):
    model = LitESPnetModel(dummy_model, config, dummy_dataset, dummy_dataset)
    train_dl = model.train_dataloader()

    assert isinstance(train_dl, torch.utils.data.DataLoader)
    assert train_dl.batch_size == 80
    assert train_dl.num_workers == 0
    assert hasattr(train_dl.collate_fn, "int_pad_value")
    assert train_dl.collate_fn.int_pad_value == -1


def test_val_dataloader_structure(config, dummy_model, dummy_dataset):
    model = LitESPnetModel(dummy_model, config, dummy_dataset, dummy_dataset)
    val_dl = model.val_dataloader()

    assert isinstance(val_dl, torch.utils.data.DataLoader)
    assert val_dl.batch_size == 4
    assert val_dl.num_workers == 0
    assert hasattr(val_dl.collate_fn, "int_pad_value")
    assert val_dl.collate_fn.int_pad_value == -1


def test_configure_optimizers(config, dummy_model, dummy_dataset):
    model = LitESPnetModel(dummy_model, config, dummy_dataset, dummy_dataset)
    result = model.configure_optimizers()
    assert "optimizer" in result
    assert isinstance(result["optimizer"], torch.optim.Optimizer)
    assert "lr_scheduler" in result


def test_state_dict_delegates_to_model(config, dummy_dataset):
    dummy_model = torch.nn.Module()
    dummy_model.state_dict = mock.MagicMock(return_value={"foo": torch.tensor([42.0])})

    model = LitESPnetModel(dummy_model, config, dummy_dataset, dummy_dataset)
    result = model.state_dict()

    dummy_model.state_dict.assert_called_once()
    assert isinstance(result, dict)
    assert "foo" in result


def test_load_state_dict_delegates_to_model(config, dummy_dataset):
    dummy_model = torch.nn.Module()
    dummy_model.load_state_dict = mock.MagicMock(return_value="load_result")

    model = LitESPnetModel(dummy_model, config, dummy_dataset, dummy_dataset)
    dummy_state = {"foo": torch.tensor([1.0])}
    result = model.load_state_dict(dummy_state, strict=False)

    dummy_model.load_state_dict.assert_called_once_with(dummy_state, strict=False)
    assert result == "load_result"
