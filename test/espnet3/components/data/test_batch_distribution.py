"""Batch distribution preserves the selected per-rank or global sample budget."""

import json
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from espnet2.samplers.category_balanced_sampler import CategoryBalancedSampler
from espnet3.components.data import dataloader as dataloader_module
from espnet3.components.data.dataloader import DataLoaderBuilder
from espnet3.components.data.dataset import CombinedDataset
from espnet3.utils.config_utils import load_default_config

SEQUENCE_FACTORY = "espnet2.iterators.sequence_iter_factory.SequenceIterFactory"
BATCHES = [list(range(5)), list(range(5, 10)), list(range(10, 13))]


def _config(distribution=None, mode="train", **factory_options):
    loader = {
        "iter_factory": {
            "_target_": SEQUENCE_FACTORY,
            "shuffle": False,
            "batches": {},
            **factory_options,
        }
    }
    if distribution is not None:
        loader["batch_distribution"] = distribution
    return OmegaConf.create({"dataloader": {mode: loader}})


def _dataset():
    return CombinedDataset(
        datasets=[[{"value": index} for index in range(20)]],
        transforms=[(None, None)],
    )


def _builder(config, num_device=1):
    return DataLoaderBuilder(_dataset(), config, None, num_device, epoch=0)


def _patch_ranks(monkeypatch, rank, world_size):
    dist = torch.distributed
    monkeypatch.setattr(dist, "is_initialized", lambda: world_size > 1)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)

    def gather(output, value):
        output[:] = [value] * world_size

    monkeypatch.setattr(dist, "all_gather_object", gather)


def _values(loader):
    # Mocked ranks test distribution; real process tests also exercise epoch sync.
    return [batch["value"].tolist() for batch in loader.generate()]


@pytest.mark.parametrize("distribution", [None, "per_gpu", "global"])
@pytest.mark.parametrize("mode", ["train", "valid"])
def test_single_device_distribution(monkeypatch, distribution, mode):
    """One rank receives all samples unchanged, including singleton batches."""
    batches = [[0], [1, 2, 3], [4, 5]]
    monkeypatch.setattr(
        dataloader_module, "build_batch_sampler", lambda **kwargs: batches
    )
    _patch_ranks(monkeypatch, 0, 1)
    assert _values(_builder(_config(distribution, mode)).build(mode)) == batches


@pytest.mark.parametrize("distribution", [None, "per_gpu", "global"])
@pytest.mark.parametrize("world_size", [2, 3])
def test_iter_factory_distribution(monkeypatch, distribution, world_size):
    """Whole-batch striding and element striding produce distinct sample sets."""
    monkeypatch.setattr(
        dataloader_module, "build_batch_sampler", lambda **kwargs: BATCHES
    )
    outputs = []
    for rank in range(world_size):
        _patch_ranks(monkeypatch, rank, world_size)
        outputs.append(
            _values(_builder(_config(distribution), world_size).build("train"))
        )
    if distribution == "global":
        for step, batch in enumerate(BATCHES):
            rank_batches = [output[step] for output in outputs]
            assert rank_batches == [
                batch[rank::world_size] for rank in range(world_size)
            ]
            assert sorted(item for part in rank_batches for item in part) == batch
    else:
        kept = BATCHES[: len(BATCHES) - len(BATCHES) % world_size]
        assert outputs == [kept[rank::world_size] for rank in range(world_size)]


@pytest.mark.parametrize("distribution", [None, "per_gpu", "global"])
def test_batch_smaller_than_world_size(monkeypatch, distribution):
    """Small batches remain valid per GPU but cannot yield empty global ranks."""
    monkeypatch.setattr(
        dataloader_module, "build_batch_sampler", lambda **kwargs: [[0], [1]]
    )
    for rank in range(2):
        _patch_ranks(monkeypatch, rank, 2)
        builder = _builder(_config(distribution), 2)
        if distribution == "global":
            with pytest.raises(RuntimeError, match="batch.*1.*world_size=2"):
                builder.build("train")
        else:
            assert _values(builder.build("train")) == [[rank]]


@pytest.mark.parametrize("distribution", [None, "per_gpu"])
def test_standard_loader_backward_compatibility(distribution):
    """The optional setting is not forwarded to torch's DataLoader."""
    config = _config(distribution)
    config.dataloader.train.iter_factory = None
    config.dataloader.train.batch_size = 2
    loader = _builder(config).build("train")
    assert len(loader) == 10
    assert next(iter(loader))["value"].tolist() == [0, 1]


@pytest.mark.parametrize("distribution", [None, "per_gpu", "global"])
def test_dynamic_sampler_distribution(tmp_path, monkeypatch, distribution):
    """Real numel batches preserve their samples and budget under rank splitting."""
    shapes = tmp_path / "feats_shape"
    shapes.write_text("".join(f"{index} {index + 1},1\n" for index in range(20)))
    config = _config(distribution)
    config.dataloader.train.iter_factory.batches = {
        "type": "numel",
        "batch_size": 1,
        "batch_bins": 60,
        "shape_files": [str(shapes)],
        "min_batch_size": 3,
    }
    batches = list(
        dataloader_module.build_batch_sampler(
            **OmegaConf.to_container(config.dataloader.train.iter_factory.batches)
        )
    )
    for rank in range(2):
        _patch_ranks(monkeypatch, rank, 2)
        observed = _values(_builder(config, 2).build("train"))
        if distribution == "global":
            expected = [[int(key) for key in batch[rank::2]] for batch in batches]
        else:
            kept = batches[: len(batches) - len(batches) % 2]
            expected = [[int(key) for key in batch] for batch in kept[rank::2]]
        assert observed == expected


def test_default_training_config_preserves_per_gpu():
    """Loading the template explicitly retains the existing batching default."""
    config = load_default_config(
        "training.yaml", default_package="egs3.TEMPLATE.esp2_asr"
    )
    assert config.dataloader.train.batch_distribution == "per_gpu"
    assert config.dataloader.valid.batch_distribution == "per_gpu"


@pytest.mark.parametrize("distribution", ["unknown", None, 1])
def test_invalid_distribution_is_rejected(distribution):
    """An explicitly invalid value must not select another batching mode."""
    config = _config()
    config.dataloader.train.batch_distribution = distribution
    with pytest.raises(ValueError, match="batch_distribution"):
        _builder(config).build("train")


def test_global_requires_iter_factory():
    """Standard DataLoader does not expose the shared batch plan to split."""
    config = _config("global")
    config.dataloader.train.iter_factory = None
    with pytest.raises(RuntimeError, match="global.*iter_factory"):
        _builder(config).build("train")


@pytest.mark.parametrize("rank", [0, 1])
def test_global_accepts_custom_iter_factory(monkeypatch, rank):
    """Custom factories receive split batches without a class-name restriction."""
    monkeypatch.setattr(
        dataloader_module, "build_batch_sampler", lambda **kwargs: BATCHES
    )
    _patch_ranks(monkeypatch, rank, 2)
    config = _config("global")
    config.dataloader.train.iter_factory._target_ = (
        "test.espnet3.components.data.test_dataloader_builder.DummyIterFactory"
    )
    loader = _builder(config, 2).build("train")
    assert list(loader.generate()) == [batch[rank::2] for batch in BATCHES]


@pytest.mark.parametrize("epoch", [0, 1])
def test_global_category_iter_factory(tmp_path, monkeypatch, epoch):
    """Category batches split correctly at startup and after epoch regeneration."""
    categories = tmp_path / "category2utt"
    categories.write_text(
        "".join(
            f"category{category} "
            + " ".join(str(index) for index in range(category * 5, (category + 1) * 5))
            + "\n"
            for category in range(4)
        )
    )
    sampler_args = {
        "category2utt_file": str(categories),
        "batch_size": 4,
        "min_batch_size": 2,
        "num_batches": None,
        "distributed": True,
    }
    initial_batches = list(CategoryBalancedSampler(**sampler_args, epoch=1))
    expected_batches = list(CategoryBalancedSampler(**sampler_args, epoch=epoch + 1))
    monkeypatch.setattr(
        dataloader_module, "build_batch_sampler", lambda **kwargs: initial_batches
    )
    config = _config(
        "global", sampler_args=sampler_args, batch_type="catbel", _convert_="all"
    )
    config.dataloader.train.iter_factory._target_ = (
        "espnet2.iterators.category_iter_factory.CategoryIterFactory"
    )
    for rank in range(2):
        _patch_ranks(monkeypatch, rank, 2)
        builder = _builder(config, 2)
        builder.epoch = epoch
        assert _values(builder.build("train")) == [
            [int(index) for index in batch[rank::2]] for batch in expected_batches
        ]


@pytest.mark.parametrize("sampler_args", [None, {"distributed": False}])
def test_global_rejects_unsharded_category_regeneration(monkeypatch, sampler_args):
    """Reject category settings that would duplicate batches from the next epoch."""
    monkeypatch.setattr(
        dataloader_module, "build_batch_sampler", lambda **kwargs: BATCHES
    )
    _patch_ranks(monkeypatch, 0, 2)
    config = _config("global", sampler_args=sampler_args)
    config.dataloader.train.iter_factory._target_ = (
        "espnet2.iterators.category_iter_factory.CategoryIterFactory"
    )
    with pytest.raises(RuntimeError, match="sampler_args.distributed=True"):
        _builder(config, 2).build("train")


def test_global_rejects_rank_specific_dataset():
    """Reject sharding before a rank's dataset can be replaced with another shard."""
    dataset = _dataset()
    inner = type("Shard", (), {"total_shards": 2})()
    dataset.datasets.append(inner)
    builder = DataLoaderBuilder(dataset, _config("global"), None, 2, epoch=0)
    with pytest.raises(RuntimeError, match="global.*unsharded"):
        builder.build("train")


def test_global_rejects_batch_reshuffling():
    """Local reshuffling after splitting would change global batch membership."""
    with pytest.raises(RuntimeError, match="global.*shuffle_within_batch"):
        _builder(_config("global", shuffle_within_batch=True)).build("train")


def test_global_requires_initialized_distributed_group(monkeypatch):
    """A configured multi-device job cannot silently run as one rank."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="global.*initialized"):
        _builder(_config("global"), num_device=2).build("train")


def test_global_uses_runtime_world_size(monkeypatch):
    """One device per node still splits over all ranks in the process group."""
    monkeypatch.setattr(
        dataloader_module, "build_batch_sampler", lambda **kwargs: BATCHES
    )
    _patch_ranks(monkeypatch, 1, 2)
    loader = _builder(_config("global"), num_device=1).build("train")
    assert _values(loader) == [batch[1::2] for batch in BATCHES]


def test_global_rejects_different_rank_plans(monkeypatch):
    """Different sample plans must fail rather than silently change the batch."""
    monkeypatch.setattr(
        dataloader_module, "build_batch_sampler", lambda **kwargs: BATCHES
    )
    _patch_ranks(monkeypatch, 0, 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value: output.__setitem__(slice(None), [value, "different"]),
    )
    with pytest.raises(RuntimeError, match="global.*identical"):
        _builder(_config("global"), 2).build("train")


def _distributed_worker(rank, directory, distribution, scenario):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{directory}/process_group",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        batches = BATCHES
        if scenario == "different" and rank == 1:
            batches = list(reversed(batches))
        elif scenario == "small":
            batches = [[0], [1]]
        with patch.object(
            dataloader_module, "build_batch_sampler", return_value=batches
        ):
            builder = _builder(_config(distribution), num_device=2)
            try:
                loader = builder.build("train")
                result = [batch["value"].tolist() for batch in loader]
            except RuntimeError as error:
                result = str(error)
        (directory / f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.execution_timeout(300)
@pytest.mark.parametrize(
    "distribution,scenario",
    [
        ("per_gpu", "normal"),
        ("global", "normal"),
        ("global", "different"),
        ("global", "small"),
    ],
)
def test_distribution_with_real_process_group(tmp_path, distribution, scenario):
    """Real Gloo ranks agree on steps, reject mismatches, and terminate together."""
    context = torch.multiprocessing.spawn(
        _distributed_worker,
        args=(tmp_path, distribution, scenario),
        nprocs=2,
        join=False,
    )
    try:
        while not context.join(timeout=1):
            pass
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
    results = [
        json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)
    ]
    if scenario == "normal":
        if distribution == "global":
            assert results == [
                [batch[rank::2] for batch in BATCHES] for rank in range(2)
            ]
        else:
            assert results == [[BATCHES[0]], [BATCHES[1]]]
    else:
        message = "identical" if scenario == "different" else "world_size=2"
        assert all(isinstance(result, str) and message in result for result in results)
