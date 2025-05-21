import os
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.train.collate_fn import CommonCollateFn
from espnet3.trainer.model import LitESPnetModel
from espnet3.trainer.sampler import MappedSamplerWrapper


# Dummy dataset for testing
class DummyDataset:
    def __init__(self, path=None):
        self.data = [
            {
                "id": "utt_a",
                "audio": np.random.random(16000).astype(np.float32),
                "text": np.array([1, 2, 3]).astype(np.int64),
            },
            {
                "id": "utt_b",
                "audio": np.random.random(32000).astype(np.float32),
                "text": np.array([1, 2, 3, 4]).astype(np.int64),
            },
            {
                "id": "utt_c",
                "audio": np.random.random(48000).astype(np.float32),
                "text": np.array([1, 2, 3, 4, 5]).astype(np.int64),
            },
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        uid = data["id"]
        return uid, {"audio": data["audio"], "text": data["text"]}


@pytest.fixture
def external_dummy_dataset_config():
    return {
        "_target_": "espnet3.data.DataOrganizer",
        "train": [
            {
                "name": "dummy_train",
                "dataset": {"_target_": "test.espnet3.test_model.DummyDataset"},
            }
        ],
        "valid": [
            {
                "name": "dummy_valid",
                "dataset": {"_target_": "test.espnet3.test_model.DummyDataset"},
            }
        ],
        "test": [
            {
                "name": "dummy_test",
                "dataset": {"_target_": "test.espnet3.test_model.DummyDataset"},
            }
        ],
    }


# ---------------------- Dummy Model ----------------------


@pytest.fixture
def dummy_model():
    model = torch.nn.Linear(1, 1)
    model.forward = lambda **kwargs: (
        torch.tensor(0.123),
        {"loss": torch.tensor(0.123)},
        torch.tensor(1.0),
    )
    return model


# ---------------------- Shape file helper ----------------------


def write_shape_file(tmp_path, lines, filename="shape.txt"):
    path = tmp_path / filename
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    return str(path)


# ---------------------- A系：Sampler Wrapper ----------------------


class DummySampler:
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def generate(self, seed):
        return list(self.batches)


def test_wrapper_basic(tmp_path):
    shape_path = write_shape_file(tmp_path, ["utt_a 123", "utt_b 456", "utt_c 789"])
    base = DummySampler([["utt_a", "utt_b"], ["utt_c"]])
    wrapper = MappedSamplerWrapper(base, [shape_path])
    batches = list(wrapper)
    assert batches == [(0, 1), (2,)]


def test_wrapper_duplicate_uttid(tmp_path):
    path = write_shape_file(tmp_path, ["utt_a 1", "utt_a 2"])
    with pytest.raises(ValueError, match="Duplicate uttid"):
        MappedSamplerWrapper(DummySampler([["utt_a"]]), [path])


def test_wrapper_missing_uttid(tmp_path):
    path = write_shape_file(tmp_path, ["utt_a 1"])
    wrapper = MappedSamplerWrapper(DummySampler([["utt_b"]]), [path])
    with pytest.raises(KeyError, match="utt_b"):
        _ = list(wrapper)


def test_wrapper_multiple_files(tmp_path):
    s1 = write_shape_file(tmp_path, ["utt_a 1", "utt_b 2"], "shape_1.txt")
    s2 = write_shape_file(tmp_path, ["utt_c 3"], "shape_2.txt")
    wrapper = MappedSamplerWrapper(DummySampler([["utt_a", "utt_c"]]), [s1, s2])
    assert list(wrapper) == [(0, 2)]


# ---------------------- B系：DataLoader / Model Tests ----------------------


def test_dataloader_with_dataorganizer_config(
    tmp_path, dummy_model, external_dummy_dataset_config
):
    config = OmegaConf.create(
        {
            "expdir": str(tmp_path / "exp"),
            "dataset": external_dummy_dataset_config,
            "dataloader": {
                "train": {"batch_size": 2, "num_workers": 0},
                "valid": {"batch_size": 2, "num_workers": 0},
            },
        }
    )
    model = LitESPnetModel(dummy_model, config)
    dl = model.train_dataloader()
    batch = next(iter(dl))
    assert isinstance(batch, (list, tuple))
    assert "audio" in batch[1] and "text" in batch[1]
    assert hasattr(dl.collate_fn, "int_pad_value")
    assert dl.collate_fn.int_pad_value == -1


def test_training_step_runs(tmp_path, dummy_model, external_dummy_dataset_config):
    config = OmegaConf.create(
        {
            "expdir": str(tmp_path / "exp"),
            "dataset": external_dummy_dataset_config,
            "dataloader": {
                "train": {"batch_size": 2, "num_workers": 0},
                "valid": {"batch_size": 2, "num_workers": 0},
            },
        }
    )
    model = LitESPnetModel(dummy_model, config)
    dl = model.train_dataloader()
    batch = next(iter(dl))
    out = model.training_step(batch, 0)
    assert np.allclose(out.item(), np.array([0.123]))


# ---------------------- C系：IterFactory 条件分岐 ----------------------


class DummyIterFactory:
    def __init__(self, dataset, batches):
        self.dataset = dataset
        self.sampler = batches
        self.collate_fn = CommonCollateFn(int_pad_value=-1)

    def build_iter(self, epoch, shuffle=False):
        return torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler,
            collate_fn=self.collate_fn,
        )


class DummySamplerWithShape:
    def __init__(
        self,
        shape_files,
        batch_bins: int = 3,
        min_batch_size: int = 1,
    ):
        self.batch_bins = batch_bins
        self.min_batch_size = min_batch_size
        self.shape_files = shape_files

        # utt2shape: dict[utt_id -> List[int]]
        utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]
        base_utt2shape = utt2shapes[0]

        # shapeの整合性をチェック
        for s in utt2shapes[1:]:
            if set(s) != set(base_utt2shape):
                raise RuntimeError("utt_id mismatch across shape files")

        keys = sorted(base_utt2shape, key=lambda k: base_utt2shape[k][0])

        # バッチ分割
        self.batch_list = []
        current_batch = []
        current_bin = 0
        for key in keys:
            length = sum(d[key][0] for d in utt2shapes)
            if (
                current_bin + length > batch_bins
                and len(current_batch) >= min_batch_size
            ):
                self.batch_list.append(tuple(current_batch))
                current_batch = []
                current_bin = 0

            current_batch.append(key)
            current_bin += length

        if len(current_batch) > 0:
            self.batch_list.append(tuple(current_batch))

        total_utts = sum(len(batch) for batch in self.batch_list)
        assert total_utts == len(keys)

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self):
        for batch in self.batch_list:
            yield tuple(batch)


def test_iter_factory_with_shape_file(
    tmp_path, dummy_model, external_dummy_dataset_config
):
    shape_file = write_shape_file(tmp_path, ["utt_a 1", "utt_b 2", "utt_c 3"])
    config = OmegaConf.create(
        {
            "expdir": str(tmp_path / "exp"),
            "dataloader": {
                "train": {
                    "iter_factory": {
                        "_target_": "test.espnet3.test_model.DummyIterFactory",
                        "batches": {
                            "_target_": "test.espnet3.test_model.DummySamplerWithShape",
                            "shape_files": [shape_file],
                        },
                    }
                },
                "valid": {
                    "iter_factory": {
                        "_target_": "test.espnet3.test_model.DummyIterFactory",
                        "batches": {
                            "_target_": "test.espnet3.test_model.DummySamplerWithShape",
                            "shape_files": [shape_file],
                        },
                    }
                },
            },
            "dataset": external_dummy_dataset_config,
        }
    )
    model = LitESPnetModel(dummy_model, config)
    dl = model.train_dataloader()
    batch = next(iter(dl))
    assert len(batch[1]["audio"]) == 2
