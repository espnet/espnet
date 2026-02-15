"""Dataset classes for ESPnet3."""

import copy
from abc import ABC
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch.utils.data.dataset import Dataset


def do_nothing_transform(*x):
    """Return input as-is.

    Args:
        x: Any object.

    Returns:
        The input object unchanged.
    """
    if len(x) == 1:
        return x[0]
    else:
        return x


class CombinedDataset:
    """Combines multiple datasets into a single unified dataset-like interface.

    This class supports seamless access to multiple datasets as if they were one.
    Each dataset can be paired with a transform and a global preprocessor, which are
    applied sequentially to each sample. It also supports optional UID handling for
    ESPnet-style preprocessing.

    CombinedDataset supports two indexing modes:
        * Numeric mode (default): every underlying dataset accepts integer indices
          and the combined dataset behaves like a contiguous sequence.
        * String mode: if any dataset requires string-based utterance IDs, the
          organizer builds a lookup table mapping every UID to its source dataset
          while preserving DataLoader-friendly integer access.

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
    Note:
        At initialization, the first sample from each dataset is passed through
        its associated transform to check that all datasets produce dictionaries
        with the same set of keys. This ensures consistency across the combined dataset.
        An `AssertionError` is raised if the keys differ.

    Raises:
        IndexError: If a requested index is outside the range of the combined dataset.
        ValueError: If index is a non-integer string that none of the underlying
            datasets accept as an utterance ID.
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
        self.transforms = []
        self.lengths = [len(ds) for ds in datasets]
        self.cumulative_lengths = []
        self.use_espnet_preprocessor = use_espnet_preprocessor

        for transform, preprocessor in transforms:
            if transform is None:
                transform = do_nothing_transform
            if preprocessor is None:
                preprocessor = do_nothing_transform
            assert callable(transform), "transform must be callable."
            assert callable(preprocessor), "preprocessor must be callable."
            self.transforms.append((transform, preprocessor))

        total = 0
        for length in self.lengths:
            total += length
            self.cumulative_lengths.append(total)

        self._string_index_mode = False
        self._uid_to_dataset: Dict[str, Tuple[int, Any]] = {}
        self._dataset_supports_int: List[bool] = []
        self._dataset_key_lists: List[Optional[List[str]]] = []

        self._initialize_index_mode()

        # Check the first sample from all dataset to ensure they all have the same keys
        sample_keys = None
        for i, (dataset, transform) in enumerate(zip(self.datasets, self.transforms)):
            if len(dataset) == 0:
                continue  # Skip empty datasets

            reference_key = self._select_reference_key_for_dataset(i)
            sample = transform[0](copy.deepcopy(dataset[reference_key]))
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
        has_sharded = any(
            isinstance(dataset, ShardedDataset) for dataset in self.datasets
        )
        if has_sharded and not all(
            isinstance(dataset, ShardedDataset) for dataset in self.datasets
        ):
            raise RuntimeError(
                "If any dataset is a subclass of ShardedDataset,"
                " then all dataset should be a subclass of ShardedDataset."
            )
        if has_sharded:
            num_shards_set = {
                getattr(dataset, "num_shards", None) for dataset in self.datasets
            }
            world_shard_size_set = {
                getattr(dataset, "world_shard_size", None) for dataset in self.datasets
            }
            if None in num_shards_set or None in world_shard_size_set:
                raise RuntimeError(
                    "ShardedDataset requires num_shards and world_shard_size to be set."
                )
            if len(num_shards_set) != 1 or len(world_shard_size_set) != 1:
                raise RuntimeError(
                    "All sharded datasets must share the same num_shards and "
                    "world_shard_size."
                )
            self.num_shards = num_shards_set.pop()
            self.world_shard_size = world_shard_size_set.pop()

        # This flag will be overrode by ESPnetLightningModule.
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
        if self._string_index_mode:
            return self._getitem_string_mode(idx)

        if isinstance(idx, str):
            try:
                numerical_idx = int(idx)
            except (ValueError, TypeError):
                return self._getitem_by_utterance_id(idx)
            else:
                idx = numerical_idx

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

    def _getitem_by_utterance_id(self, uid: str):
        if self._string_index_mode:
            return self._getitem_string_mode(uid)

        last_error = None
        for dataset, (transform, preprocessor) in zip(self.datasets, self.transforms):
            try:
                sample = dataset[uid]
            except (KeyError, TypeError, ValueError, IndexError) as err:
                last_error = err
                continue

            transformed = transform(sample)
            if self.use_espnet_preprocessor:
                transformed = preprocessor(uid, transformed)
            else:
                transformed = preprocessor(transformed)

            if self.use_espnet_collator:
                return uid, transformed
            return transformed

        raise ValueError(
            f"Utterance ID '{uid}' is not supported by the underlying datasets."
        ) from last_error

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

        if self._string_index_mode:
            uid, dataset_idx, dataset_key = self._resolve_string_mode_index(idx)
            dataset = self.datasets[dataset_idx]
            return dataset.get_text(dataset_key)

        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                ds_idx = idx if i == 0 else idx - self.cumulative_lengths[i - 1]
                return self.datasets[i].get_text(ds_idx)

    # ------------------------------------------------------------------
    # Internal helpers for string-index mode
    # ------------------------------------------------------------------
    def _initialize_index_mode(self):
        """Determine whether datasets should be accessed via string keys."""
        def supports_integer_index(dataset):
            try:
                dataset[0]
            except Exception:
                return False
            else:
                return True

        self._dataset_supports_int = []
        for dataset in self.datasets:
            self._dataset_supports_int.append(supports_integer_index(dataset))

        all_integer_addressable = all(self._dataset_supports_int)
        self._dataset_key_lists = [None] * len(self.datasets)

        if all_integer_addressable:
            return

        self._string_index_mode = True

        for dataset_idx, dataset in enumerate(self.datasets):
            if self._dataset_supports_int[dataset_idx]:
                continue
            keys = self._collect_string_keys(dataset)
            self._dataset_key_lists[dataset_idx] = keys
            self._register_dataset_keys(dataset_idx, keys)

    def _collect_string_keys(self, dataset):
        if isinstance(dataset, Mapping):
            keys_iter = dataset.keys()
        elif hasattr(dataset, "keys") and callable(getattr(dataset, "keys")):
            keys_iter = dataset.keys()
        else:
            try:
                keys_iter = iter(dataset)
            except TypeError as err:
                raise TypeError(
                    "Datasets with string indices must be iterable to expose keys."
                ) from err

        keys = list(keys_iter)
        for key in keys:
            if not isinstance(key, str):
                raise TypeError(
                    "Datasets operating in string-index mode must provide string keys."
                )
        return keys

    def _register_dataset_keys(self, dataset_idx: int, keys: List[str]):
        for key in keys:
            if key in self._uid_to_dataset:
                raise ValueError(
                    f"Duplicate utterance ID '{key}' detected across datasets."
                )
            self._uid_to_dataset[key] = (dataset_idx, key)

    def _select_reference_key_for_dataset(self, dataset_idx: int):
        if not self._string_index_mode or self._dataset_supports_int[dataset_idx]:
            return 0

        keys = self._dataset_key_lists[dataset_idx]
        if not keys:
            raise RuntimeError("Unable to locate reference key for dataset.")
        return keys[0]

    def _resolve_string_mode_index(self, idx):
        if isinstance(idx, int):
            if idx < 0:
                raise IndexError("Index out of range in CombinedDataset")
            dataset_idx = 0
            for i, cum_len in enumerate(self.cumulative_lengths):
                if idx < cum_len:
                    dataset_idx = i
                    break
            else:
                raise IndexError("Index out of range in CombinedDataset")

            ds_idx = idx if dataset_idx == 0 else idx - self.cumulative_lengths[dataset_idx - 1]
            if self._dataset_supports_int[dataset_idx]:
                uid = str(idx)
                dataset_key = ds_idx
            else:
                keys = self._dataset_key_lists[dataset_idx]
                if keys is None:
                    raise RuntimeError("String dataset keys are not initialized.")
                dataset_key = keys[ds_idx]
                uid = dataset_key
            return uid, dataset_idx, dataset_key

        if isinstance(idx, str):
            try:
                dataset_idx, dataset_key = self._uid_to_dataset[idx]
            except KeyError as err:
                raise ValueError(
                    f"Utterance ID '{idx}' is not supported by the underlying datasets."
                ) from err
            return idx, dataset_idx, dataset_key

        raise TypeError("Index must be an integer or string utterance ID.")

    def _getitem_string_mode(self, idx):
        uid, dataset_idx, dataset_key = self._resolve_string_mode_index(idx)
        dataset = self.datasets[dataset_idx]
        transform, preprocessor = self.transforms[dataset_idx]

        sample = dataset[dataset_key]
        transformed = transform(sample)
        if self.use_espnet_preprocessor:
            transformed = preprocessor(uid, transformed)
        else:
            transformed = preprocessor(transformed)

        if self.use_espnet_collator:
            return uid, transformed
        return transformed

    def shard(self, shard_idx: int):
        """Return a sharded version of the combined dataset.

        This is used when handling large datasets that are split into shards
        for efficiency and distributed processing.
        All datasets must be subclasses of
        `espnet3.components.data.dataset.ShardedDataset` and implement
        a `shard()` method.

        Args:
            shard_idx (int): Index of the shard to retrieve.

        Returns:
            CombinedDataset: A new CombinedDataset containing the sharded datasets.

        Raises:
            RuntimeError: If any dataset does not support sharding.
        """
        if not all(isinstance(dataset, ShardedDataset) for dataset in self.datasets):
            raise RuntimeError(
                "All dataset should be the subclass of "
                "espnet3.components.data.dataset.ShardedDataset."
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
        if transform is None:
            transform = do_nothing_transform
        assert callable(transform), "transform must be callable."
        if preprocessor is None:
            preprocessor = do_nothing_transform
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

    This interface is used when datasets are split into shards for parallel or
    distributed data loading. Any dataset subclassing `ShardedDataset` must
    implement the `shard()` method.

    Attributes:
        num_shards (int): Total number of shards in the dataset.
        world_shard_size (int): Expected distributed world size when sharding.

    Note:
        - This class is intended to be used with `CombinedDataset` in ESPnet.
        - All datasets combined must subclass `ShardedDataset` if sharding is used.

    Example:
        >>> class MyDataset(ShardedDataset):
        ...     def __init__(self):
        ...         self.num_shards = 8
        ...         self.world_shard_size = 4
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
