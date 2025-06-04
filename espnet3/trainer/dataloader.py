from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
import copy

import torch
import numpy as np

from espnet2.train.collate_fn import CommonCollateFn
from espnet3.data.dataset import ShardedDataset

from typing import Any, Union


def update_shard(config: Union[dict, list], shard_idx: int) -> Union[dict, list]:
    """
    Recursively update all strings in the config that contain "{shard_idx}" 
    with the actual shard index.

    Args:
        config: A nested config dictionary or list (from OmegaConf.to_container).
        shard_idx: The shard index to format into strings.

    Returns:
        A new config dict/list with formatted strings.
    """
    if isinstance(config, dict):
        return {
            k: update_shard(v, shard_idx)
            for k, v in config.items()
        }
    elif isinstance(config, list):
        return [update_shard(v, shard_idx) for v in config]
    elif isinstance(config, str) and "{shard_idx}" in config:
        return config.format(shard_idx=shard_idx)
    else:
        return config


class DataLoaderBuilder:
    def __init__(self, dataset, config, collate_fn, num_device: int, epoch: int):
        self.dataset = dataset
        self.config = config
        self.collate_fn = collate_fn
        self.num_device = num_device
        self.epoch = epoch

    def build(self, mode: str):
        mode_config = getattr(self.config.dataloader, mode, DictConfig({}))

        config = copy.copy(mode_config)
        if hasattr(config, "multiple_iterator") and config.multiple_iterator:
            return self._build_multiple_iterator(config)
        if config.iter_factory is not None:
            return self._build_iter_factory(config.iter_factory)
        return self._build_standard_dataloader(config)

    def _build_standard_dataloader(self, dataloader_config, dataset=None):
        if dataset is None:
            dataset = self.dataset

        config = OmegaConf.to_container(dataloader_config, resolve=True)
        sampler = batch_sampler = None

        if isinstance(self.collate_fn, CommonCollateFn):
            dataset.add_uid = True

        if hasattr(self.config, "sampler"):
            sampler = instantiate(self.config.sampler, dataset)
        if hasattr(self.config, "batch_sampler"):
            batch_sampler = instantiate(self.config.batch_sampler, dataset)

        assert not (sampler and batch_sampler), \
            "Cannot specify both sampler and batch_sampler"
        config.pop("dataset", None)

        # Remove default config for espnet's data loader
        config.pop("iter_factory")

        return torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            **config,
        )

    def _build_iter_factory(self, factory_config, dataset=None):
        if dataset is None:
            dataset = self.dataset
        
        batches = instantiate(factory_config.pop("batches"))

        if self.num_device > 1:
            batches = list(batches)
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            for batch in batches:
                if len(batch) < world_size:
                    raise RuntimeError(
                        f"The batch-size must be equal or more than world_size: {len(batch)} < {world_size}"
                    )
            batches = [batch[rank::world_size] for batch in batches]

        iter_factory = instantiate(
            factory_config,
            dataset,
            batches=batches
        )

        return iter_factory.build_iter(self.epoch, shuffle=False)

    def _build_multiple_iterator(self, factory_config):
        assert self.dataset.multiple_iterator, \
            "All dataset must be a subclass of espnet3.data.dataset.ShardedDataset"

        assert hasattr(factory_config, "num_shards"), \
            "When using multiple iterator, please specify the number of shards."
        
        num_shards = factory_config.num_shards
        shuffle = factory_config.get("shuffle", False)
        seed = self.config.get("seed", 0)
        rng = np.random.RandomState(self.epoch + seed)
        shard_idx = rng.choice(num_shards) if shuffle else self.epoch % num_shards

        dataset = self.dataset.shard(shard_idx)
        
        if factory_config.iter_factory is not None:
            # update shape files
            iter_factory_config = OmegaConf.to_container(
                factory_config.iter_factory, resolve=True
            )
            iter_factory_config = update_shard(iter_factory_config, shard_idx)
            return self._build_iter_factory(iter_factory_config, dataset)
        else:
            factory_config.pop("num_shards")
            factory_config.pop("multiple_iterator")
            return self._build_standard_dataloader(factory_config, dataset)

