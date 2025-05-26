from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

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

        if mode_config.multiple_iterator:
            return self._build_multiple_iterator(mode_config)
        if mode_config.iter_factory is not None:
            return self._build_iter_factory(mode_config.iter_factory)
        return self._build_standard_dataloader(mode_config)

    def _build_standard_dataloader(self, dataloader_config):
        config = OmegaConf.to_container(dataloader_config, resolve=True)
        sampler = batch_sampler = None

        if isinstance(self.collate_fn, CommonCollateFn):
            self.dataset.add_uid = True

        if hasattr(self.config, "sampler"):
            sampler = instantiate(self.config.sampler, self.dataset)
        if hasattr(self.config, "batch_sampler"):
            batch_sampler = instantiate(self.config.batch_sampler, self.dataset)

        assert not (sampler and batch_sampler), \
            "Cannot specify both sampler and batch_sampler"
        config.pop("dataset", None)

        return torch.utils.data.DataLoader(
            self.dataset,
            sampler=sampler,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            **config,
        )

    def _build_iter_factory(self, factory_config, dataset=None):
        if dataset is None:
            dataset = self.dataset

        iter_factory = instantiate(factory_config, dataset)

        if self.num_device > 1:
            batches = list(iter_factory.sampler)
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            for batch in batches:
                if len(batch) < world_size:
                    raise RuntimeError(
                        f"The batch-size must be equal or more than world_size: {len(batch)} < {world_size}"
                    )
            iter_factory.batches = [batch[rank::world_size] for batch in batches]

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
        
        # update shape files
        iter_factory_config = OmegaConf.to_container(
            factory_config.iter_factory, resolve=True
        )
        iter_factory_config = update_shard(iter_factory_config, shard_idx)
        return self._build_iter_factory(factory_config.iter_factory, dataset)

