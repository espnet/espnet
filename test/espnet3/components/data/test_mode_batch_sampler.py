"""A per-mode `batch_sampler` replaces batch_size/shuffle for that loader only."""

import torch
from omegaconf import OmegaConf

from espnet3.components.data.dataloader import DataLoaderBuilder
from espnet3.components.data.dataset import CombinedDataset


class EvenOddBatches:
    def __init__(self, sampler, batch_size, drop_last, dataset):
        self.sampler = sampler
        self.batches = [
            list(range(start, len(dataset), 2))[:batch_size] for start in (0, 1)
        ]
        self.drop_last = drop_last

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _loader(mode_config):
    dataset = CombinedDataset(
        datasets=[[{"value": index} for index in range(10)]],
        transforms=[(None, None)],
    )
    config = OmegaConf.create({"dataloader": {"train": mode_config}})
    return DataLoaderBuilder(dataset, config, None, 1, epoch=0).build("train")


def test_mode_batch_sampler_gets_the_batch_size_and_owns_the_order():
    loader = _loader(
        {
            "iter_factory": None,
            "batch_size": 3,
            "shuffle": True,
            "drop_last": True,
            "batch_sampler": {"_target_": f"{__name__}.EvenOddBatches"},
        }
    )
    assert loader.batch_sampler.drop_last is True
    assert isinstance(loader.batch_sampler.sampler, torch.utils.data.RandomSampler)
    assert [batch["value"].tolist() for batch in loader] == [[0, 2, 4], [1, 3, 5]]
