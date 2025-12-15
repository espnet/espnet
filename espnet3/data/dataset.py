"""Dataset classes for ESPnet3."""

import copy
from abc import ABC
from typing import Any, Callable, List, Tuple

from torch.utils.data.dataset import Dataset


class CombinedDataset:
    """Combines multiple datasets into a single unified dataset-like interface.

    This class supports seamless access to multiple datasets as if they were one.
    Each dataset can be paired with a transform and a global preprocessor, which are
    applied sequentially to each sample. It also supports optional UID handling for
    ESPnet-style preprocessing.

    Args:
        datasets (List[Any]): A list of dataset instances. Each must implement
            `__getitem__` and `__len__`.
        transforms (List[Tuple[Callable, Callable]]): A list of
            (transform, preprocessor) tuples. Each pair corresponds to the matching
            dataset in `datasets`.
            - `transform(sample)` is applied first.
            - Then `preprocessor(uid, sample)` or `preprocessor(sample)` is applied,
              depending on `use_espnet_preprocessor`.
        use_espnet_preprocessor (bool): If True, applies the preprocessor as
            `preprocessor(uid, sample)`. This is used for ESPnet `AbsPreprocessor`
            compatible pipelines.

    Attributes:
        get_text_available (bool): True if all datasets implement `get_text(idx)`.
        multiple_iterator (bool): True if any dataset is a subclass of `ShardedDataset`.

    Note:
        At initialization, the first sample from each dataset is passed through
        its associated transform to check that all datasets produce dictionaries
        with the same set of keys. This ensures consistency across the combined dataset.
        An `AssertionError` is raised if the keys differ.

    Raises:
        IndexError: If a requested index is outside the range of the combined dataset.
        ValueError: If index is a non-integer string or cannot be cast to int.
        RuntimeError: If `get_text()` or `shard()` is called but not supported.
        AssertionError: If output keys from different datasets are inconsistent.

    Example:
        >>> dataset = CombinedDataset(
        ...     datasets=[ds1, ds2],
        ...     transforms=[
        ...         (transform1, preprocessor),
        ...         (transform2, preprocessor),
        ...     ],
        ...     use_espnet_preprocessor=True
        ... )
        >>> sample = dataset[5]
        >>> print(sample["text"])
    """

    def __init__(
        self,
        datasets: List[Any],
        transforms: List[Tuple[Callable, Callable]],
        use_espnet_preprocessor: bool = False,
    ):
        """Initialize CombinedDataset object."""
        self.datasets = datasets
        self.transforms = transforms
        self.lengths = [len(ds) for ds in datasets]
        self.cumulative_lengths = []
        self.use_espnet_preprocessor = use_espnet_preprocessor

        total = 0
        for length in self.lengths:
            total += length
            self.cumulative_lengths.append(total)

        # Check the first sample from all dataset to ensure they all have the same keys
        sample_keys = None
        for i, (dataset, transform) in enumerate(zip(self.datasets, self.transforms)):
            if len(dataset) == 0:
                continue  # Skip empty datasets
            sample = transform[0](copy.deepcopy(dataset[0]))
            keys = set(sample.keys())
            if sample_keys is None:
                sample_keys = keys
            else:
                assert keys == sample_keys, (
                    f"Inconsistent output keys in dataset {i}: "
                    f"{keys} != {sample_keys}"
                )

        # Check if get_text is available
        self.get_text_available = True
        for dataset in self.datasets:
            if not hasattr(dataset, "get_text"):
                self.get_text_available = False

        # Check if dataset is a subclass of ShardedDataset.
        self.multiple_iterator = False
        for dataset in self.datasets:
            if isinstance(dataset, ShardedDataset):
                self.multiple_iterator = True
            if self.multiple_iterator and not isinstance(dataset, ShardedDataset):
                raise RuntimeError(
                    "If any dataset is a subclass of ShardedDataset,"
                    " then all dataset should be a subclass of ShardedDataset."
                )

        # This flag will be overrode by LitESPnetModel when initializing dataloader.
        self._use_espnet_collator = False

    @property
    def use_espnet_collator(self):
        """Get the flag indicating whether to use ESPnet collator."""
        return self._use_espnet_collator

    @use_espnet_collator.setter
    def use_espnet_collator(self, value: bool):
        """Set the flag indicating whether to use ESPnet collator."""
        self._use_espnet_collator = value

    def __len__(self):
        """Return the total number of samples in the combined dataset."""
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        """Retrieve and process a sample by index."""
        if isinstance(idx, str):
            try:
                idx = int(idx)
            except (ValueError, TypeError):
                raise ValueError(
                    "ESPnet-3 expects the utterance ID to be an " "integer index"
                )

        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                ds_idx = idx if i == 0 else idx - self.cumulative_lengths[i - 1]
                try:
                    sample = self.datasets[i][ds_idx]
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to access dataset at index {i} or "
                        f"item at index {ds_idx}. "
                        f"Original error: {e}"
                    ) from e

                transformed = self.transforms[i][0](sample)  # apply transform
                if self.use_espnet_preprocessor:
                    transformed = self.transforms[i][1](str(idx), transformed)
                else:
                    transformed = self.transforms[i][1](transformed)

                if self.use_espnet_collator:
                    return str(idx), transformed
                else:
                    return transformed

        raise IndexError("Index out of range in CombinedDataset")

    def get_text(self, idx):
        """Retrieve the target text string for a given index.

        This method delegates to the underlying dataset's `get_text(idx)` method.
        It is typically used for extracting text sequences for purposes such as
        training tokenizers or language models.

        Raises:
            RuntimeError: If not all datasets implement `get_text(idx)`.
        """
        if not self.get_text_available:
            raise RuntimeError(
                "Please define `get_text` function to all datasets."
                "It should receive index of data and return target text."
                "E.g., \n"
                "def get_text(self, idx):\n"
                "   return text\n"
            )

        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                ds_idx = idx if i == 0 else idx - self.cumulative_lengths[i - 1]
                return self.datasets[i].get_text(ds_idx)

    def shard(self, shard_idx: int):
        """Return a sharded version of the combined dataset.

        This is used when handling large datasets that are split into shards
        for efficiency and distributed processing (ESPnet multiple-iterator mode).
        All datasets must be subclasses of `espnet3.data.dataset.ShardedDataset`,
        and implement a `shard()` method.

        Args:
            shard_idx (int): Index of the shard to retrieve.

        Returns:
            CombinedDataset: A new CombinedDataset containing the sharded datasets.

        Raises:
            RuntimeError: If any dataset does not support sharding.
        """
        if not self.multiple_iterator:
            raise RuntimeError(
                "All dataset should be the subclass of "
                "espnet3.data.dataset.ShardedDataset."
            )
        sharded_datasets = [dataset.shard(shard_idx) for dataset in self.datasets]
        return CombinedDataset(
            sharded_datasets,
            self.transforms,
            self.use_espnet_preprocessor,
        )


class DatasetWithTransform:
    """Lightweight wrapper for applying a transform function to dataset items.

    This class wraps a dataset and applies a user-defined transform followed by a
    preprocessor function. It also supports ESPnet-style UID handling, where the
    preprocessor receives both a UID and the sample.

    Args:
        dataset (Any): A dataset implementing `__getitem__` and `__len__`.
        transform (Callable): A function applied to each sample before preprocessor.
        preprocessor (Callable): A function applied after the transform.
            If `use_espnet_preprocessor` is True, it must accept `(uid, sample)`
            as arguments. Otherwise, it must accept a single `sample`.
        use_espnet_preprocessor (bool): Whether to include the UID when calling
            the preprocessor. Required for ESPnet's `AbsPreprocessor` compatibility.

    Example:
        >>> def transform(sample):
        ...     return {
        ...         "text": sample["text"].upper()
        ...     }
        >>>
        >>> def preprocessor(uid, sample):
        ...     return {
        ...         "text": f"[uid={uid}] " + sample["text"]
        ...     }
        >>>
        >>> wrapped = DatasetWithTransform(
        ...     my_dataset,
        ...     transform,
        ...     preprocessor,
        ...     use_espnet_preprocessor=True
        ... )
        >>> uid_sample = wrapped[0]
        >>> print(uid_sample["text"])
        [uid=0] HELLO

    Raises:
        TypeError: If `preprocessor` is not callable.
        TypeError: If `transform` is not callable.
    """

    def __init__(self, dataset, transform, preprocessor, use_espnet_preprocessor=False):
        """Initialize DatasetWithTransform."""
        assert callable(transform), "transform must be callable."
        assert callable(preprocessor), "preprocessor must be callable."
        self.dataset = dataset
        self.transform = transform
        self.preprocessor = preprocessor
        self.use_espnet_preprocessor = use_espnet_preprocessor

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Retrieve and process a sample by index."""
        sample = self.dataset[idx]
        transformed = self.transform(sample)  # apply transform
        if self.use_espnet_preprocessor:
            transformed = self.preprocessor(str(idx), transformed)
        else:
            transformed = self.preprocessor(transformed)
        return transformed

    def __call__(self, idx):
        """Alias for __getitem__ to allow callable access."""
        return self.__getitem__(idx)


class ShardedDataset(ABC, Dataset):
    """Abstract base class for datasets that support sharding.

    This interface is used in ESPnet's multiple-iterator mode, where datasets are split
    into shards for parallel or distributed data loading. Any dataset subclassing
    `ShardedDataset` must implement the `shard()` method.

    Note:
        - This class is intended to be used with `CombinedDataset` in ESPnet.
        - All datasets combined must subclass `ShardedDataset` if sharding is used.

    Example:
        >>> class MyDataset(ShardedDataset):
        ...     def shard(self, idx):
        ...         return Subset(self, shard_indices[idx])

    """

    def shard(self, idx: int):
        """Return a new dataset shard corresponding to the given index.

        This method must be implemented by subclasses to return a subset of the data
        for sharded training or evaluation.

        Args:
            idx (int): The index of the shard to return.

        Returns:
            Dataset: A dataset instance representing the shard.

        Raises:
            NotImplementedError: Always in the base class. Must be overridden.
        """
        raise NotImplementedError(
            "Please implement `shard` function, "
            "which should return a `torch.utils.data.Dataset` object "
            "representing the shard corresponding to the given index."
        )
