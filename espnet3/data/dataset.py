from abc import ABC
from typing import Any, Callable, Dict, List, Tuple, Union

from torch.utils.data.dataset import Dataset

from espnet2.train.dataset import AbsDataset


class ESPnetEZDataset(AbsDataset):
    """
    A dataset class for handling ESPnet data with easy access to data information.

    This class extends the AbsDataset class and provides functionalities to
    manage a dataset and its associated metadata. It allows users to retrieve
    dataset items using unique identifiers and check for available names in
    the dataset.

    Attributes:
        dataset (Union[list, Tuple]): The dataset containing the actual data entries.
        data_info (Dict[str, callable]): A dictionary mapping attribute names to
            functions that extract those attributes from the dataset.

    Args:
        dataset (Union[list, Tuple]): The dataset from which data will be extracted.
        data_info (Dict[str, callable]): A dictionary where keys are attribute names
            and values are functions that process the dataset entries.

    Methods:
        has_name(name): Checks if the given name exists in the data_info dictionary.
        names() -> Tuple[str, ...]: Returns a tuple of all names in the data_info.
        __getitem__(uid: Union[str, int]) -> Tuple[str, Dict]: Retrieves the data
            entry corresponding to the provided unique identifier.
        __len__() -> int: Returns the total number of entries in the dataset.

    Examples:
        >>> dataset = [
            ("audio1.wav", "transcription1"),
            ("audio2.wav", "transcription2")
        ]
        >>> data_info = {
        ...     "audio": lambda x: x[0],
        ...     "transcription": lambda x: x[1]
        ... }
        >>> ez_dataset = ESPnetEZDataset(dataset, data_info)
        >>> ez_dataset.has_name("audio")
        True
        >>> ez_dataset.names()
        ('audio', 'transcription')
        >>> ez_dataset[0]
        ('0', {'audio': 'audio1.wav', 'transcription': 'transcription1'})
        >>> len(ez_dataset)
        2

    Note:
        The dataset and data_info must be provided in a compatible format to ensure
        proper functionality of the methods.
    """

    def __init__(self, dataset, data_info=None):
        self.dataset = dataset
        self.data_info = data_info

    def has_name(self, name) -> bool:
        """
        Check if the specified name exists in the dataset's data information.

        This method searches the `data_info` attribute of the dataset to determine
        if the given `name` is present as a key. It is useful for validating
        whether certain attributes or features are available in the dataset.

        Args:
            name (str): The name to search for in the dataset's data information.

        Returns:
            bool: True if the name exists in the data information; False otherwise.

        Examples:
            >>> dataset = ESPnetEZDataset(dataset=[...],
                data_info={'feature1': ..., 'feature2': ...})
            >>> dataset.has_name('feature1')
            True
            >>> dataset.has_name('feature3')
            False

        Note:
            The method performs a simple membership check using the `in` operator,
            which is efficient for dictionaries.
        """
        return name in self.data_info

    def names(self) -> Tuple[str, ...]:
        """
            A dataset class for ESPnet that handles data retrieval and management.

        This class extends the abstract dataset class to provide functionalities
        specific to the ESPnet framework. It manages a dataset and its associated
        metadata, allowing for efficient data access and manipulation.

        Attributes:
            dataset (Union[list, tuple]): The underlying dataset that contains the data.
            data_info (Dict[str, callable]): A dictionary mapping names to functions
                that process each data entry in the dataset.

        Args:
            dataset (Union[list, tuple]): The dataset to be wrapped.
            data_info (Dict[str, callable]): A dictionary where keys are the names of
                the data attributes and values are functions that extract or transform
                the data from the dataset.

        Methods:
            has_name(name: str) -> bool:
                Checks if a given name exists in the data_info.

            names() -> Tuple[str, ...]:
                Returns a tuple of all the names available in the data_info.

            __getitem__(uid: Union[str, int]) -> Tuple[str, Dict]:
                Retrieves the data entry corresponding to the provided identifier.

            __len__() -> int:
                Returns the number of entries in the dataset.

        Examples:
            >>> dataset = ESPnetEZDataset(dataset=[...],
                data_info={'feature': lambda x: x.feature, 'label': lambda x: x.label})
            >>> dataset.has_name('feature')
            True
            >>> dataset.names()
            ('feature', 'label')
            >>> entry = dataset[0]
            >>> print(entry)
            ('0', {'feature': ..., 'label': ...})
            >>> len(dataset)
            100

        Note:
            The functions provided in the data_info should be callable and should
            accept a single argument corresponding to an entry from the dataset.
        """
        return tuple(self.data_info.keys())

    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict]:
        idx = int(uid)
        return (
            str(uid),
            {k: v(self.dataset[idx]) for k, v in self.data_info.items()},
        )

    def __len__(self) -> int:
        return len(self.dataset)


class ShardedDataset(ABC, Dataset):
    def shard(self, idx: int):
        raise NotImplementedError(
            "Please implement `shard` function,"
            " which will return torch.utils.data.dataset.Dataset class"
            " based on shard index."
        )


class CombinedDataset:
    """
    Combines multiple datasets into a single dataset-like interface.

    This allows unified iteration over multiple datasets, applying their associated
    transforms and mapping a flat index to the correct dataset and item.

    Args:
        datasets (List[Any]): List of datasets implementing __getitem__ and __len__.
        transforms (List[Tuple[Callable, Callable]]): List of tuple with transform and
            preprocessor. [(transform1, preprocessor), (transform2, preprocessor)..]

    Example:
        >>> dataset = CombinedDataset([ds1, ds2], [tf1, tf2])
        >>> sample = dataset[10]

    Raises:
        IndexError: If index is out of range of the combined dataset.
    """

    def __init__(
        self,
        datasets: List[Any],
        transforms: List[Tuple[Callable, Callable]],
        add_uid: bool = False,
    ):
        self.datasets = datasets
        self.transforms = transforms
        self.lengths = [len(ds) for ds in datasets]
        self.cumulative_lengths = []
        self.add_uid = add_uid
        total = 0
        for length in self.lengths:
            total += length
            self.cumulative_lengths.append(total)

        # Check the first sample from all dataset to ensure they all have the same keys
        sample_keys = None
        for i, (dataset, transform) in enumerate(zip(self.datasets, self.transforms)):
            if len(dataset) == 0:
                continue  # Skip empty datasets
            sample = transform[0](dataset[0].copy())
            if isinstance(sample, tuple):  # (uid, data_dict)
                _, sample = sample
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
            if self.multiple_iterator and not isinstance(dataset, ShardedDataset):
                raise RuntimeError(
                    "If any dataset is a subclass of ShardedDataset,"
                    " then all dataset should be a subclass of ShardedDataset."
                )
            if isinstance(dataset, ShardedDataset):
                self.multiple_iterator = True

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        if isinstance(idx, str):
            try:
                idx = int(idx)
            except:
                raise ValueError("ESPnet-3 expext the utterance ID to be integer index")

        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                ds_idx = idx if i == 0 else idx - self.cumulative_lengths[i - 1]
                sample = self.datasets[i][ds_idx]
                transformed = self.transforms[i][0](sample)  # apply transform
                if self.add_uid:
                    transformed = self.transforms[i][1](str(idx), transformed)
                else:
                    transformed = self.transforms[i][1](transformed)
                return transformed

        raise IndexError("Index out of range in CombinedDataset")

    def get_text(self, idx):
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
        if not self.multiple_iterator:
            raise RuntimeError(
                "All dataset should be the subclass of "
                "espnet3.data.dataset.ShardedDataset."
            )
        sharded_datasets = [dataset.shard(shard_idx) for dataset in self.datasets]
        return CombinedDataset(
            sharded_datasets,
            self.transforms,
            self.add_uid,
        )


class DatasetWithTransform:
    """
    Lightweight wrapper for applying a transform function to dataset items.

    Args:
        dataset (Any): A dataset implementing __getitem__ and __len__.
        transform (Callable): A transform function applied to each sample.
        transform (Callable): A preprocess function applied to each sample.

    Example:
        >>> wrapped = DatasetWithTransform(my_dataset, my_transform)
        >>> item = wrapped[0]
    """

    def __init__(self, dataset, transform, preprocessor, add_uid=False):
        self.dataset = dataset
        self.transform = transform
        self.preprocessor = preprocessor
        self.add_uid = add_uid

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        transformed = self.transform(sample)  # apply transform
        if self.add_uid:
            transformed = self.preprocessor(str(idx), transformed)
        else:
            transformed = self.preprocessor(transformed)
        return transformed

    def __call__(self, idx):
        return self.__getitem__(idx)
