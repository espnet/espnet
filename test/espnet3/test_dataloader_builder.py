import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import BatchSampler, Sampler

from espnet3.data.data_organizer import DataOrganizer, do_nothing_transform
from espnet3.data.dataset import ShardedDataset
from espnet3.trainer.dataloader import DataLoaderBuilder
from espnet3.utils.config import load_config_with_defaults

# ===============================================================
# Test Case Summary for DataLoaderBuilder
# ===============================================================
#
# Standard PyTorch DataLoader mode
# | Test Name                          | Description                                                              | # noqa: E501
# |-----------------------------------|--------------------------------------------------------------------------| # noqa: E501
# | test_batch_sampler_only           | Uses only batch_sampler and omits batch_size and shuffle                | # noqa: E501
# | test_sampler_only                 | Uses only sampler to drive the dataloader                               | # noqa: E501
# | test_collate_fn_none              | Sets collate_fn=None to use PyTorch's default_collate                   | # noqa: E501
# | test_common_collate_fn            | Uses CommonCollateFn and checks audio/audio_lengths formatting          | # noqa: E501
# | test_custom_collate_fn            | Uses a custom collate function and checks the batch structure           | # noqa: E501
# | test_sampler_and_batch_sampler_conflict | Ensures AssertionError if both sampler and batch_sampler are defined  | # noqa: E501
#
# IterFactory Mode & YAML-based Integration
# | Test Name                          | Description                                                              | # noqa: E501
# |-----------------------------------|--------------------------------------------------------------------------| # noqa: E501
# | test_iter_factory_from_default_yaml_with_organizer | Uses a real YAML config with iter_factory and verifies batch structure | # noqa: E501
# | test_iter_factory_with_collate_fn | Confirms config-defined collate_fn takes precedence over argument       | # noqa: E501
#
# Multiple Iterator Mode (Sharded Dataset)
# | Test Name                              | Description                                                              | # noqa: E501
# |---------------------------------------|--------------------------------------------------------------------------| # noqa: E501
# | test_multiple_iterator_shard_initialization | Checks correct shard is selected based on epoch                         | # noqa: E501
# | test_multiple_iterator_epoch_shard_switching | Ensures different shard is selected as epoch changes                  | # noqa: E501
#
# Note:
# - DummyDataset and DummyShardedDataset are used to simulate real-world data layout
# - All `DataLoaderBuilder.build(mode)` modes are exercised: standard, iter_factory,
# and multiple_iterator


# -------- Dummy components for testing --------


class DummyDataset:
    def __init__(self, path=None):
        self.data = [{"audio": np.random.random(16000 * (i + 1))} for i in range(10)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DummyDatasetSameLength:
    def __init__(self, path=None):
        self.data = [{"audio": np.random.random(16000)} for i in range(10)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DummySampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class DummyBatchSampler(BatchSampler):
    def __init__(self, data_source):
        self.batches = [list(range(0, 5)), list(range(5, 10))]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def dummy_collate_fn(batch):
    return {"custom_collated": batch}


# -------- Config mocks --------


def make_standard_dataloader_config(sampler=None, batch_sampler=None, collate_fn=None):
    dataloader_config = dict(
        batch_size=2,
        shuffle=False,
        drop_last=False,
        iter_factory=None,
    )
    config = dict(
        dataloader=dict(train=dataloader_config),
        sampler=sampler,
        batch_sampler=batch_sampler,
    )
    return OmegaConf.create(config)


# -------- Standard PyTorch DataLoader mode --------


def test_batch_sampler_only():
    dataset = DummyDataset()
    config = make_standard_dataloader_config(
        batch_sampler={
            "_target_": "test.espnet3.test_dataloader_builder.DummyBatchSampler"
        }
    )
    # We don't need batch size for batch sampler
    del config.dataloader.train.batch_size
    del config.dataloader.train.shuffle
    builder = DataLoaderBuilder(dataset, config, collate_fn=None, num_device=1, epoch=0)
    loader = builder.build("train")
    assert "DummyBatchSampler" in str(loader.batch_sampler.__class__)


def test_sampler_only():
    dataset = DummyDataset()
    config = make_standard_dataloader_config(
        sampler={"_target_": "test.espnet3.test_dataloader_builder.DummySampler"}
    )
    builder = DataLoaderBuilder(dataset, config, collate_fn=None, num_device=1, epoch=0)
    loader = builder.build("train")
    assert "DummySampler" in str(loader.sampler.__class__)


def test_collate_fn_none():
    dataset = DummyDatasetSameLength()
    config = make_standard_dataloader_config()
    builder = DataLoaderBuilder(dataset, config, collate_fn=None, num_device=1, epoch=0)
    loader = builder.build("train")
    batch = next(iter(loader))
    assert isinstance(batch, dict)
    assert isinstance(batch["audio"], torch.Tensor)  # default_collate used


def test_common_collate_fn():
    from espnet2.train.collate_fn import CommonCollateFn

    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_dataloader_builder.DummyDataset"
                },
            }
        ],
    }
    config = OmegaConf.create(config)
    dataset = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["train"]),
        preprocessor=do_nothing_transform,
    )
    config = make_standard_dataloader_config()
    collate_fn = CommonCollateFn(int_pad_value=-1)
    dataset.train.use_espnet_collator = True
    builder = DataLoaderBuilder(
        dataset.train, config, collate_fn=collate_fn, num_device=1, epoch=0
    )
    loader = builder.build("train")
    batch = next(iter(loader))
    assert isinstance(batch[0], list)
    assert isinstance(batch[1], dict)
    assert "audio" in batch[1]
    assert "audio_lengths" in batch[1]


def test_custom_collate_fn():
    dataset = DummyDataset()
    config = make_standard_dataloader_config()
    builder = DataLoaderBuilder(
        dataset, config, collate_fn=dummy_collate_fn, num_device=1, epoch=0
    )
    loader = builder.build("train")
    batch = next(iter(loader))
    assert "custom_collated" in batch


def test_sampler_and_batch_sampler_conflict():
    dataset = DummyDataset()
    config = make_standard_dataloader_config(
        sampler={"_target_": "test.espnet3.test_dataloader_builder.DummySampler"},
        batch_sampler={
            "_target_": "test.espnet3.test_dataloader_builder.DummyBatchSampler"
        },
    )
    builder = DataLoaderBuilder(dataset, config, collate_fn=None, num_device=1, epoch=0)
    with pytest.raises(
        AssertionError, match="Cannot specify both sampler and batch_sampler"
    ):
        _ = builder._build_standard_dataloader(config.dataloader.train)


# -------- IterFactory Mode & YAML-based Integration --------


def test_iter_factory_from_default_yaml_with_organizer(tmp_path):
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_dataloader_builder.DummyDataset"
                },
            }
        ],
    }
    config = OmegaConf.create(config)
    dataset = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["train"]),
        preprocessor=do_nothing_transform,
    )
    yaml_text = """
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn:
        _target_: espnet2.train.collate_fn.CommonCollateFn
        int_pad_value: -1
      batches:
        shape_files:
          - test_utils/espnet3/stats/stats_dummy
        type: unsorted
        batch_size: 2
        batch_bins: 4000000
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)

    cfg = load_config_with_defaults(str(config_path))
    dataset.train.use_espnet_collator = True
    builder = DataLoaderBuilder(
        dataset=dataset.train,
        config=cfg,
        collate_fn=None,  # Defined on config
        num_device=1,
        epoch=0,
    )
    loader = builder.build("train")
    batch = next(iter(loader))
    assert len(batch[0]) == 2
    assert "audio" in batch[1]
    assert "audio_lengths" in batch[1]


def test_iter_factory_with_collate_fn(tmp_path):
    # For this case the collate_fn in configuration is used.
    # The collate_fn in DataloaderBuilder.__init__ is only used in standard dataloader.
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_dataloader_builder.DummyDataset"
                },
            }
        ],
    }
    config = OmegaConf.create(config)
    dataset = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["train"]),
        preprocessor=do_nothing_transform,
    )
    yaml_text = """
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn:
        _target_: espnet2.train.collate_fn.CommonCollateFn
        int_pad_value: -1
      batches:
        shape_files:
          - test_utils/espnet3/stats/stats_dummy
        type: unsorted
        batch_size: 2
        batch_bins: 4000000
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text)

    cfg = load_config_with_defaults(str(config_path))
    dataset.train.use_espnet_collator = True
    builder = DataLoaderBuilder(
        dataset=dataset.train,
        config=cfg,
        collate_fn=dummy_collate_fn,  # Defined on config
        num_device=1,
        epoch=0,
    )
    loader = builder.build("train")
    batch = next(iter(loader))
    print(batch)
    assert len(batch[0]) == 2
    assert "audio" in batch[1]
    assert "audio_lengths" in batch[1]


# --- Multiple Iterator Mode (Sharded Dataset) ---


# Dummy sharded dataset that returns shard-specific samples
class DummyShardedDataset(ShardedDataset):
    def __init__(self, shard_id: int = None):
        if shard_id is None:
            self.samples = []
        else:
            self.samples = [
                {"text": f"shard{shard_id}_sample0"},
                {"text": f"shard{shard_id}_sample1"},
            ]
        self.shard_id = shard_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def shard(self, idx: int):
        return DummyShardedDataset(idx)


@pytest.fixture
def dummy_multiple_iterator_dataset(tmp_path):
    # YAML-style instantiation config for 3 shards
    dataset_config = {
        "train": [
            {
                "name": "shard0",
                "dataset": {
                    "_target_": "test.espnet3.test_dataloader_builder.DummyShardedDataset",  # noqa: E501
                },
            },
        ],
        "valid": [
            {
                "name": "valid",
                "dataset": {
                    "_target_": "test.espnet3.test_dataloader_builder.DummyShardedDataset",  # noqa: E501
                },
            }
        ],
    }
    return OmegaConf.create(dataset_config)


def make_multiple_iterator_config(num_shards: int, shuffle: bool):
    return OmegaConf.create(
        {
            "dataloader": {
                "train": {
                    "multiple_iterator": True,
                    "num_shards": num_shards,
                    "shuffle": shuffle,
                    "iter_factory": None,
                    "batch_size": 1,
                    "num_workers": 0,
                }
            },
            "num_device": 1,
        }
    )


def test_multiple_iterator_shard_initialization(dummy_multiple_iterator_dataset):
    organizer = DataOrganizer(
        train=instantiate(dummy_multiple_iterator_dataset.train),
        valid=instantiate(dummy_multiple_iterator_dataset.valid),
    )
    config = make_multiple_iterator_config(num_shards=3, shuffle=False)
    builder = DataLoaderBuilder(
        dataset=organizer.train,
        config=config,
        collate_fn=None,
        num_device=1,
        epoch=0,
    )
    loader = builder.build("train")
    batch = next(iter(loader))
    assert "text" in batch
    assert batch["text"][0].startswith("shard0_")


@pytest.mark.parametrize(
    "epoch,expected_shard", [(0, "shard0"), (1, "shard1"), (2, "shard2")]
)
def test_multiple_iterator_epoch_shard_switching(
    dummy_multiple_iterator_dataset, epoch, expected_shard
):
    organizer = DataOrganizer(
        train=instantiate(dummy_multiple_iterator_dataset["train"]),
        valid=instantiate(dummy_multiple_iterator_dataset["valid"]),
    )
    config = make_multiple_iterator_config(num_shards=3, shuffle=False)
    builder = DataLoaderBuilder(
        dataset=organizer.train,
        config=config,
        collate_fn=None,
        num_device=1,
        epoch=epoch,
    )
    loader = builder.build("train")
    batch = next(iter(loader))
    assert expected_shard in batch["text"][0]
