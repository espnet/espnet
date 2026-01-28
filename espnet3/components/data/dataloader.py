"""DataLoader builder for ESPnet3 trainer."""

import copy

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet2.samplers.build_batch_sampler import build_batch_sampler


class DataLoaderBuilder:
    """Builder class for constructing training and validation DataLoaders in ESPnet3.

    This class provides a unified interface for setting up PyTorch or ESPnet-specific
    DataLoaders based on the configuration. It supports advanced features such as:

    - Custom collate functions (e.g., CommonCollateFn)
    - Sequence-based batch sampling with batch_bins or batch_size
    - Dynamic handling of DDP-compatible iteration strategies

    Typically used by ESPnetLightningModule to instantiate training and validation
    DataLoaders with consistent behavior across single-GPU, multi-GPU, and distributed
    training.

    Args:
        dataset (torch.utils.data.Dataset): Dataset instance wrapped by DataLoader.
        config (DictConfig): Full training configuration (e.g., from OmegaConf).
        collate_fn (Callable): Function to collate individual samples into mini-batches.
        num_device (int): Number of devices in use (e.g., GPUs or nodes).
        epoch (int): Current epoch number. Used to reseed samplers deterministically.

    Example:
        builder = DataLoaderBuilder(
            dataset=train_dataset,
            config=config,
            collate_fn=collate_fn,
            num_device=4,
            epoch=3
        )
        train_loader = builder.build(mode="train")
    """

    def __init__(self, dataset, config, collate_fn, num_device: int, epoch: int):
        """Initialize DataLoaderBuilder object."""
        self.dataset = dataset
        self.config = config
        self.collate_fn = collate_fn
        self.num_device = num_device
        self.epoch = epoch

    def build(self, mode: str):
        """Build and return a DataLoader for the specified mode ("train" or "valid").

        This method supports two modes of operation depending on the configuration:
        (1) **ESPnet-style iterator**: Custom sampling and iteration logic.
        (2) **Standard PyTorch DataLoader**: Simpler use-case with fixed batch size.

        The selection depends on whether `config.dataloader.<mode>.iter_factory` is
        defined.

        Args:
            mode (str): One of {"train", "valid"} indicating which dataloader to build.

        Returns:
            torch.utils.data.DataLoader: Configured DataLoader instance for training or
                validation.

        Config-driven branching logic:
            Case 1: ESPnet SequenceIterFactory is used
            Example config:
            ```
            dataloader:
            train:
                iter_factory:
                _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
                shuffle: true
                collate_fn: ${dataloader.collate_fn}
                batches:
                    _target_: espnet2.samplers.build_batch_sampler.build_batch_sampler
                    type: numel
                    shape_files:
                    batch_bins: 4000000
                    min_batch_size: 2
            ```

            Case 2: Standard PyTorch DataLoader is used
            Example config:
            ```
            dataloader:
            train:
              iter_factory: # Set iter_factory to None
              batch_size: 4
              num_workers: 2
              shuffle: true
            ```

        Raises:
            ValueError: If the provided mode is neither "train" nor "valid".
        """
        mode_config = getattr(self.config.dataloader, mode, DictConfig({}))

        config = copy.copy(mode_config)
        if hasattr(config, "multiple_iterator"):
            raise RuntimeError(
                "ESPnet3 does not support multiple_iterator. "
                "If you need sharding, select a shard explicitly "
                "(e.g., point the dataset/shape files to split.*) "
                "and use a standard iter_factory."
            )
        target = None
        if config.iter_factory is not None:
            target = getattr(config.iter_factory, "_target_", None)
            if target is None and isinstance(config.iter_factory, dict):
                target = config.iter_factory.get("_target_")
        if target == "espnet2.iterators.multiple_iter_factory.MultipleIterFactory":
            raise RuntimeError(
                "MultipleIterFactory is not supported in ESPnet3. "
                "Use a standard iter_factory (Sequence/Chunk/Category*) and "
                "select a single shard explicitly if needed."
            )
        if config.iter_factory is not None:
            factory_config = OmegaConf.to_container(config.iter_factory, resolve=True)
            return self._build_iter_factory(factory_config)
        return self._build_standard_dataloader(config)

    def _build_standard_dataloader(self, dataloader_config, dataset=None):
        if dataset is None:
            dataset = self.dataset

        config = OmegaConf.to_container(dataloader_config, resolve=True)
        sampler = batch_sampler = None

        if hasattr(self.config, "sampler"):
            sampler = instantiate(self.config.sampler, dataset)
        if hasattr(self.config, "batch_sampler"):
            batch_sampler = instantiate(self.config.batch_sampler, dataset)

        assert not (
            sampler and batch_sampler
        ), "Cannot specify both sampler and batch_sampler"
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

        batches = build_batch_sampler(**factory_config.pop("batches"))

        if self.num_device > 1:
            batches = list(batches)
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            for batch in batches:
                if len(batch) < world_size:
                    raise RuntimeError(
                        "The batch-size must be equal or more than world_size:"
                        f"{len(batch)} < {world_size}"
                    )
            batches = [batch[rank::world_size] for batch in batches]

        iter_factory = instantiate(factory_config, dataset, batches=batches)

        return iter_factory.build_iter(self.epoch, shuffle=False)
