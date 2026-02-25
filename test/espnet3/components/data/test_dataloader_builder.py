import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import BatchSampler, Sampler

from espnet3.components.data.dataloader import DataLoaderBuilder
from espnet3.components.data.dataset import ShardedDataset
from espnet3.utils.config_utils import load_config_with_defaults

# | Test Name                                         | Description                                                    | # noqa: E501
# |--------------------------------------------------|----------------------------------------------------------------| # noqa: E501
# | test_batch_sampler_only                          | Builds loader using batch_sampler without batch_size/shuffle   | # noqa: E501
# | test_sampler_only                                | Builds loader using sampler                                    | # noqa: E501
# | test_collate_fn_none                             | Uses default_collate when collate_fn is None                   | # noqa: E501
# | test_common_collate_fn                           | Uses CommonCollateFn and checks audio/audio_lengths in batch   | # noqa: E501
# | test_custom_collate_fn                           | Uses custom collate function and checks batch structure        | # noqa: E501
# | test_sampler_and_batch_sampler_conflict          | Raises when sampler and batch_sampler are both set             | # noqa: E501
# | test_iter_factory_from_default_yaml_with_organizer | Builds iter_factory from YAML and validates batch            | # noqa: E501
# | test_iter_factory_with_collate_fn                | Prefers config-defined collate_fn over argument                | # noqa: E501
# | test_iter_factory_drops_tail_batches_for_ddp     | Drops tail batches for DDP to match world size                 | # noqa: E501
# | test_multiple_iterator_shard_initialization      | Selects correct shard at epoch 0                               | # noqa: E501
# | test_multiple_iterator_epoch_shard_switching     | Switches shard with epoch index                                | # noqa: E501

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
# Note:
# - DummyDataset and DummyDatasetSameLength are used to simulate real-world data layout
# - All `DataLoaderBuilder.build(mode)` modes are exercised: standard, iter_factory

DUMMY_DATASET_TARGET = (
    "test.espnet3.components.data.test_dataloader_builder." "DummyDataset"
)
DUMMY_SAME_LENGTH_DATASET_TARGET = (
    "test.espnet3.components.data.test_dataloader_builder." "DummyDatasetSameLength"
)
DUMMY_SAMPLER_TARGET = (
    "test.espnet3.components.data.test_dataloader_builder." "DummySampler"
)
DUMMY_BATCH_SAMPLER_TARGET = (
    "test.espnet3.components.data.test_dataloader_builder." "DummyBatchSampler"
)
DUMMY_SHARDED_DATASET_TARGET = (
    "test.espnet3.components.data.test_dataloader_builder." "DummyShardedDataset"
)
DUMMY_MISSING_SHARD_TARGET = (
    "test.espnet3.components.data.test_dataloader_builder."
    "DummyMissingShardMethod"
)


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


class DummyShardedDataset(ShardedDataset):
    def __init__(self, shard_id: int = 0, num_shards: int = 2, world_shard_size: int = 1):
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.world_shard_size = world_shard_size
        self.data = [
            {"text": f"shard{shard_id}_sample0"},
            {"text": f"shard{shard_id}_sample1"},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def shard(self, idx: int):
        return DummyShardedDataset(
            shard_id=idx,
            num_shards=self.num_shards,
            world_shard_size=self.world_shard_size,
        )


class DummyMissingShardMethod:
    def __init__(self, num_shards=2, world_shard_size=1):
        self.num_shards = num_shards
        self.world_shard_size = world_shard_size

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError


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


def make_dataset_config(dataset_target, dataset_kwargs=None):
    dataset_kwargs = dataset_kwargs or {}
    config = {
        "_target_": "espnet3.components.data.data_organizer.DataOrganizer",
        "train": [
            {
                "name": "train_dummy",
                "dataset": {"_target_": dataset_target, **dataset_kwargs},
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {"_target_": dataset_target, **dataset_kwargs},
            }
        ],
    }
    return OmegaConf.create(config)


def build_organizer(dataset_target, dataset_kwargs=None):
    config = make_dataset_config(dataset_target, dataset_kwargs=dataset_kwargs)
    return instantiate(config)


def build_builder(dataset, config, collate_fn, num_device, epoch):
    return DataLoaderBuilder(
        dataset=dataset,
        config=config,
        collate_fn=collate_fn,
        num_device=num_device,
        epoch=epoch,
    )


# -------- Standard PyTorch DataLoader mode --------


def test_batch_sampler_only():
    organizer = build_organizer(DUMMY_DATASET_TARGET)
    config = make_standard_dataloader_config(
        batch_sampler={"_target_": DUMMY_BATCH_SAMPLER_TARGET}
    )
    # We don't need batch size for batch sampler
    del config.dataloader.train.batch_size
    del config.dataloader.train.shuffle
    builder = build_builder(
        organizer.train, config, collate_fn=None, num_device=1, epoch=0
    )
    loader = builder.build("train")
    assert "DummyBatchSampler" in str(loader.batch_sampler.__class__)


def test_sampler_only():
    organizer = build_organizer(DUMMY_DATASET_TARGET)
    config = make_standard_dataloader_config(sampler={"_target_": DUMMY_SAMPLER_TARGET})
    builder = build_builder(
        organizer.train, config, collate_fn=None, num_device=1, epoch=0
    )
    loader = builder.build("train")
    assert "DummySampler" in str(loader.sampler.__class__)


def test_collate_fn_none():
    organizer = build_organizer(DUMMY_SAME_LENGTH_DATASET_TARGET)
    config = make_standard_dataloader_config()
    builder = build_builder(
        organizer.train, config, collate_fn=None, num_device=1, epoch=0
    )
    loader = builder.build("train")
    batch = next(iter(loader))
    assert isinstance(batch, dict)
    assert isinstance(batch["audio"], torch.Tensor)  # default_collate used


def test_common_collate_fn():
    from espnet2.train.collate_fn import CommonCollateFn

    organizer = build_organizer(DUMMY_DATASET_TARGET)
    config = make_standard_dataloader_config()
    collate_fn = CommonCollateFn(int_pad_value=-1)
    organizer.train.use_espnet_collator = True
    builder = build_builder(
        organizer.train, config, collate_fn=collate_fn, num_device=1, epoch=0
    )
    loader = builder.build("train")
    batch = next(iter(loader))
    assert isinstance(batch[0], list)
    assert isinstance(batch[1], dict)
    assert "audio" in batch[1]
    assert "audio_lengths" in batch[1]


def test_custom_collate_fn():
    organizer = build_organizer(DUMMY_DATASET_TARGET)
    config = make_standard_dataloader_config()
    builder = build_builder(
        organizer.train, config, collate_fn=dummy_collate_fn, num_device=1, epoch=0
    )
    loader = builder.build("train")
    batch = next(iter(loader))
    assert "custom_collated" in batch


def test_sampler_and_batch_sampler_conflict():
    organizer = build_organizer(DUMMY_DATASET_TARGET)
    config = make_standard_dataloader_config(
        sampler={"_target_": DUMMY_SAMPLER_TARGET},
        batch_sampler={"_target_": DUMMY_BATCH_SAMPLER_TARGET},
    )
    builder = build_builder(
        organizer.train, config, collate_fn=None, num_device=1, epoch=0
    )
    with pytest.raises(
        AssertionError, match="Cannot specify both sampler and batch_sampler"
    ):
        _ = builder._build_standard_dataloader(config.dataloader.train)


# -------- IterFactory Mode & YAML-based Integration --------


def test_iter_factory_from_default_yaml_with_organizer(tmp_path):
    organizer = build_organizer(DUMMY_DATASET_TARGET)
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
    organizer.train.use_espnet_collator = True
    builder = build_builder(
        dataset=organizer.train,
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
    organizer = build_organizer(DUMMY_DATASET_TARGET)
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
    organizer.train.use_espnet_collator = True
    builder = build_builder(
        dataset=organizer.train,
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


@pytest.mark.parametrize("flag", [True, False])
def test_multiple_iterator_is_rejected(flag):
    organizer = build_organizer(DUMMY_DATASET_TARGET)
    config = make_standard_dataloader_config()
    config.dataloader.train.multiple_iterator = flag
    builder = build_builder(
        organizer.train, config, collate_fn=None, num_device=1, epoch=0
    )
    with pytest.raises(RuntimeError, match="multiple_iterator"):
        builder.build("train")


def _collect_shard_ids(loader):
    shard_ids = set()
    for batch in loader:
        text = batch["text"][0] if isinstance(batch, dict) else batch[0]["text"]
        shard_ids.add(text.split("_")[0])
        if len(shard_ids) >= 4:
            break
    return shard_ids


def _first_shard_id(loader):
    batch = next(iter(loader))
    text = batch["text"][0] if isinstance(batch, dict) else batch[0]["text"]
    return text.split("_")[0]


def test_sharded_dataset_single_gpu_multiple_shards():
    organizer = build_organizer(
        DUMMY_SHARDED_DATASET_TARGET, dataset_kwargs={"num_shards": 2, "world_shard_size": 1}
    )
    config = make_standard_dataloader_config()
    config.dataloader.train.batch_size = 1
    builder = build_builder(
        organizer.train, config, collate_fn=None, num_device=1, epoch=0
    )
    loader = builder.build("train")
    shard_ids = _collect_shard_ids(loader)
    assert shard_ids == {"shard0"}


@pytest.mark.parametrize(
    "world_size,num_shards,rank,expected",
    [
        (2, 2, 0, {"shard0"}),
        (2, 2, 1, {"shard1"}),
        (2, 4, 0, {"shard0"}),
        (2, 4, 1, {"shard1"}),
    ],
)
def test_sharded_dataset_multi_gpu_assignment(
    monkeypatch, world_size, num_shards, rank, expected
):
    def _get_world_size():
        return world_size

    def _get_rank():
        return rank

    monkeypatch.setattr(torch.distributed, "get_world_size", _get_world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", _get_rank)

    organizer = build_organizer(
        DUMMY_SHARDED_DATASET_TARGET,
        dataset_kwargs={"num_shards": num_shards, "world_shard_size": world_size},
    )
    config = make_standard_dataloader_config()
    config.dataloader.train.batch_size = 1
    builder = build_builder(
        organizer.train,
        config,
        collate_fn=None,
        num_device=world_size,
        epoch=0,
    )
    loader = builder.build("train")
    shard_ids = _collect_shard_ids(loader)
    assert shard_ids == expected


def test_sharded_dataset_multi_gpu_rotates_with_epoch(monkeypatch):
    def _get_world_size():
        return 2

    def _get_rank():
        return 0

    monkeypatch.setattr(torch.distributed, "get_world_size", _get_world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", _get_rank)

    organizer = build_organizer(
        DUMMY_SHARDED_DATASET_TARGET, dataset_kwargs={"num_shards": 4, "world_shard_size": 2}
    )
    config = make_standard_dataloader_config()
    config.dataloader.train.batch_size = 1

    builder_epoch0 = build_builder(
        organizer.train, config, collate_fn=None, num_device=2, epoch=0
    )
    first_epoch0 = _first_shard_id(builder_epoch0.build("train"))

    builder_epoch1 = build_builder(
        organizer.train, config, collate_fn=None, num_device=2, epoch=1
    )
    first_epoch1 = _first_shard_id(builder_epoch1.build("train"))

    assert first_epoch0 == "shard0"
    assert first_epoch1 == "shard2"


@pytest.mark.parametrize(
    "epoch,expected_rank0,expected_rank1",
    [
        (0, "shard0", "shard1"),
        (1, "shard2", "shard3"),
        (2, "shard0", "shard1"),
        (3, "shard2", "shard3"),
    ],
)
def test_sharded_dataset_multi_gpu_rotates_epochs_0_to_3(
    monkeypatch, epoch, expected_rank0, expected_rank1
):
    def _get_world_size():
        return 2

    monkeypatch.setattr(torch.distributed, "get_world_size", _get_world_size)

    organizer = build_organizer(
        DUMMY_SHARDED_DATASET_TARGET,
        dataset_kwargs={"num_shards": 4, "world_shard_size": 2},
    )
    config = make_standard_dataloader_config()
    config.dataloader.train.batch_size = 1

    def _first_shard_for_rank(rank):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
        builder = build_builder(
            organizer.train, config, collate_fn=None, num_device=2, epoch=epoch
        )
        return _first_shard_id(builder.build("train"))

    assert _first_shard_for_rank(0) == expected_rank0
    assert _first_shard_for_rank(1) == expected_rank1


@pytest.mark.parametrize("num_shards", [1, 3])
def test_sharded_dataset_invalid_shard_count(monkeypatch, num_shards):
    def _get_world_size():
        return 2

    def _get_rank():
        return 0

    monkeypatch.setattr(torch.distributed, "get_world_size", _get_world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", _get_rank)

    organizer = build_organizer(
        DUMMY_SHARDED_DATASET_TARGET, dataset_kwargs={"num_shards": num_shards, "world_shard_size": 2}
    )
    config = make_standard_dataloader_config()
    config.dataloader.train.batch_size = 1
    builder = build_builder(
        organizer.train, config, collate_fn=None, num_device=2, epoch=0
    )
    with pytest.raises(RuntimeError, match="num_shards must be divisible by world_size"):
        builder.build("train")


def test_sharded_dataset_missing_shard_method():
    organizer = build_organizer(
        DUMMY_MISSING_SHARD_TARGET, dataset_kwargs={"num_shards": 2, "world_shard_size": 1}
    )
    config = make_standard_dataloader_config()
    builder = build_builder(
        organizer.train, config, collate_fn=None, num_device=1, epoch=0
    )
    with pytest.raises(RuntimeError, match="shard\\(\\) is not implemented"):
        builder.build("train")


def test_sharded_dataset_world_size_mismatch(monkeypatch):
    def _get_world_size():
        return 2

    def _get_rank():
        return 0

    monkeypatch.setattr(torch.distributed, "get_world_size", _get_world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", _get_rank)

    organizer = build_organizer(
        DUMMY_SHARDED_DATASET_TARGET, dataset_kwargs={"num_shards": 2, "world_shard_size": 1}
    )
    config = make_standard_dataloader_config()
    builder = build_builder(
        organizer.train, config, collate_fn=None, num_device=2, epoch=0
    )
    with pytest.raises(RuntimeError, match="world_shard_size must match"):
        builder.build("train")
