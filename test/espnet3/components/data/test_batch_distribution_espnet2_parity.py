"""Global batch distribution reproduces ESPnet2's distributed batches exactly."""

import argparse
import random

import pytest
import torch
from omegaconf import OmegaConf

from espnet2.tasks.abs_task import IteratorOptions
from espnet2.tasks.asr import ASRTask
from espnet3.components.data.dataloader import DataLoaderBuilder
from espnet3.components.data.dataset import CombinedDataset

NUM_UTTS = 200
SEED = 7


def _patch_ranks(monkeypatch, rank, world_size):
    dist = torch.distributed
    monkeypatch.setattr(dist, "is_initialized", lambda: world_size > 1)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda output, value: output.__setitem__(slice(None), [value] * world_size),
    )


@pytest.fixture
def shape_file(tmp_path):
    # A few very long utterances make numel/length samplers emit batches
    # smaller than world_size unless min_batch_size is raised.
    rng = random.Random(0)
    lengths = [
        rng.randint(8000, 20000) if rng.random() < 0.05 else rng.randint(100, 1500)
        for _ in range(NUM_UTTS)
    ]
    path = tmp_path / "feats_shape"
    path.write_text("".join(f"{i} {n},80\n" for i, n in enumerate(lengths)))
    return path


def _espnet2_batches(monkeypatch, shape_file, batches, world_size, rank, epoch, train):
    _patch_ranks(monkeypatch, rank, world_size)
    monkeypatch.setattr(ASRTask, "build_dataset", classmethod(lambda cls, *a: None))
    args = argparse.Namespace(
        fold_length=batches.get("fold_lengths", []),
        sort_in_batch="descending",
        sort_batch="descending",
        drop_last_iter=False,
        min_batch_size=1,
        seed=SEED,
        shuffle_within_batch=False,
        num_workers=0,
        ngpu=0,
    )
    options = IteratorOptions(
        preprocess_fn=None,
        collate_fn=None,
        data_path_and_name_and_type=[(str(shape_file.parent / "wav.scp"), "x", "y")],
        shape_files=[str(shape_file)],
        batch_size=batches["batch_size"],
        batch_bins=batches["batch_bins"],
        batch_type=batches["type"],
        max_cache_size=0.0,
        max_cache_fd=0,
        allow_multi_rates=False,
        distributed=world_size > 1,
        num_batches=None,
        num_iters_per_epoch=None,
        train=train,
    )
    factory = ASRTask.build_sequence_iter_factory(args, options, "train")
    loader = factory.build_iter(epoch, shuffle=train)
    return [[int(key) for key in batch] for batch in loader.batch_sampler]


def _espnet3_batches(monkeypatch, shape_file, batches, world_size, rank, epoch, train):
    _patch_ranks(monkeypatch, rank, world_size)
    mode = "train" if train else "valid"
    config = OmegaConf.create(
        {
            "dataloader": {
                mode: {
                    "batch_distribution": "global",
                    "iter_factory": {
                        "_target_": (
                            "espnet2.iterators.sequence_iter_factory."
                            "SequenceIterFactory"
                        ),
                        "shuffle": train,
                        "seed": SEED,
                        "batches": {
                            **batches,
                            "shape_files": [str(shape_file)],
                            "sort_in_batch": "descending",
                            "sort_batch": "descending",
                        },
                    },
                }
            }
        }
    )
    dataset = CombinedDataset(
        datasets=[[{"value": i} for i in range(NUM_UTTS)]],
        transforms=[(None, None)],
    )
    # ESPnet2 epochs start at 1; DataLoaderBuilder takes Lightning's 0-based epoch.
    builder = DataLoaderBuilder(dataset, config, None, world_size, epoch=epoch - 1)
    return [batch["value"].tolist() for batch in builder.build(mode).generate()]


@pytest.mark.parametrize(
    "batches",
    [
        {"type": "numel", "batch_bins": 1_200_000, "batch_size": 1},
        {"type": "numel", "batch_bins": 300_000, "batch_size": 1},
        {"type": "length", "batch_bins": 15_000, "batch_size": 1},
        {"type": "folded", "batch_bins": 0, "batch_size": 16, "fold_lengths": [800]},
        {"type": "sorted", "batch_bins": 0, "batch_size": 8},
        {"type": "unsorted", "batch_bins": 0, "batch_size": 8},
    ],
    ids=lambda b: f"{b['type']}-{b['batch_bins'] or b['batch_size']}",
)
@pytest.mark.parametrize("world_size", [2, 3, 8])
@pytest.mark.parametrize("train", [True, False], ids=["train", "valid"])
def test_global_matches_espnet2(monkeypatch, shape_file, batches, world_size, train):
    """Same sampler config yields the same per-rank batches as ESPnet2."""
    for epoch in [1, 2] if train else [1]:
        for rank in range(world_size):
            assert _espnet3_batches(
                monkeypatch, shape_file, batches, world_size, rank, epoch, train
            ) == _espnet2_batches(
                monkeypatch, shape_file, batches, world_size, rank, epoch, train
            ), f"epoch={epoch} rank={rank}"
