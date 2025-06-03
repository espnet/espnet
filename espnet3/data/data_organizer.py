from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from datasets import load_dataset  # HuggingFace Datasets
from hydra.utils import instantiate
from tqdm import tqdm

from espnet2.train.preprocessor import AbsPreprocessor
from espnet3.data.dataset import ShardedDataset


@dataclass
class DatasetConfig:
    """
    Configuration class for dataset metadata and construction.

    This class encapsulates the necessary fields to define and instantiate a dataset.
    Used with Hydra to allow modular and flexible configuration via YAML or
    dictionaries.

    Attributes:
        name (str): Name identifier for the dataset.
        path (Optional[str]): Optional path or ID required for dataset instantiation.
        dataset (Dict[str, Any]): Python path to the dataset class to be instantiated
            via Hydra.
        transform (Optional[Callable]): Callable to transform each sample after loading.

    Example:
        >>> cfg_dict = {
        ...     "name": "custom",
            ...     "dataset": {
            ...         "_target_: "my_project.datasets.MyDataset",
            ...     },
        ...     "transform": "my_project.transforms.uppercase_transform"
        ... }
        >>> config = DatasetConfig.from_dict(cfg_dict)
    """

    name: str
    path: Optional[str] = None
    dataset: Dict[str, Any] = None  # Module path to dataset class
    transform: Optional[Dict[str, Any]] = None


class CombinedDataset:
    """
    Combines multiple datasets into a single dataset-like interface.

    This allows unified iteration over multiple datasets, applying their associated
    transforms and mapping a flat index to the correct dataset and item.

    Args:
        datasets (List[Any]): List of datasets implementing __getitem__ and __len__.
        transforms (List[Callable[[dict], dict]]): List of transform functions
            for each dataset.

    Example:
        >>> dataset = CombinedDataset([ds1, ds2], [tf1, tf2])
        >>> sample = dataset[10]

    Raises:
        IndexError: If index is out of range of the combined dataset.
    """

    def __init__(
        self,
        datasets: List[Any],
        transforms: List[Callable[[dict], dict]],
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
            sample = dataset[0]
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
                raise RuntimeError("If any dataset is a subclass of ShardedDataset," \
                    " then all dataset should be a subclass of ShardedDataset.")
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
                if self.add_uid:
                    return (str(idx), self.transforms[i]((str(idx), sample)))
                else:
                    return self.transforms[i](sample)
        raise IndexError("Index out of range in CombinedDataset")

    def get_text(self, idx):
        if not self.get_text_available:
            raise RuntimeError("Please define `get_text` function to all datasets." \
            "It should receive index of data and return target text." \
            "E.g., \n" \
            "def get_text(self, idx):\n" \
            "   return text\n")
        
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                ds_idx = idx if i == 0 else idx - self.cumulative_lengths[i - 1]
                return self.datasets[i].get_text(ds_idx)

    def shard(self, shard_idx: int):
        if not self.multiple_iterator:
            raise RuntimeError("All dataset should be the subclass of " \
                "espnet3.data.dataset.ShardedDataset.")
        sharded_datasets = [
            dataset.shard(shard_idx)
            for dataset in self.datasets
        ]
        return CombinedDataset(
            sharded_datasets,
            self.transforms,
            self.add_uid,
        )


def get_wrapped_transform(is_espnet_preprocessor, t, p):
    def transform_espnet(x):
        return p(*t(x))

    def transform(x):
        return p(t(x))

    if is_espnet_preprocessor:
        return transform_espnet
    else:
        return transform


def do_nothing_transform(x):
    return x


class DataOrganizer:
    """
    Organizes training, validation, and test datasets into a unified interface.

    Automatically instantiates datasets and applies optional transform and preprocessor
    logic per dataset.

    Args:
        config (Dict[str, Any]): Dictionary with keys 'train', 'valid',
            and optionally 'test'.
            - 'train' and 'valid' must be lists of dataset configuration dictionaries.
            - 'test' should be a dictionary of key to dataset config.
        preprocessor (Optional[Callable[[dict], dict]]): A global preprocessor function
            applied after each dataset's transform.

    Attributes:
        train (CombinedDataset): Combined dataset constructed from training configs.
        valid (CombinedDataset): Combined dataset constructed from validation configs.
        test_sets (Dict[str, DatasetWithTransform]): Mapping from test dataset names
            to wrapped datasets.

    Note:
        Raises AssertionError if 'train' or 'valid' is not present in the config.

    Example:
        >>> organizer = DataOrganizer(config_dict, preprocessor=preproc_fn)
        >>> sample = organizer.train[0]
    """

    def __init__(
        self,
        train: List[Dict[str, Any]] = None,
        valid: List[Dict[str, Any]] = None,
        test: Optional[List[Dict[str, Any]]] = None,
        preprocessor: Optional[Callable[[dict], dict]] = None,
    ):
        self.preprocessor = preprocessor or do_nothing_transform
        is_espnet_preprocessor = isinstance(self.preprocessor, AbsPreprocessor)

        def build_dataset_list(cfg_list):
            datasets = []
            transforms = []
            for cfg in tqdm(cfg_list):
                dataset = cfg.dataset
                if hasattr(cfg, "transform"):
                    transform = cfg.transform
                else:
                    transform = do_nothing_transform

                wrapped_transform = get_wrapped_transform(
                    is_espnet_preprocessor,
                    transform,
                    self.preprocessor,
                )
                datasets.append(dataset)
                transforms.append(wrapped_transform)
            return CombinedDataset(datasets, transforms, add_uid=is_espnet_preprocessor)

        self.train = None
        self.valid = None

        if train is not None:
            self.train = build_dataset_list(train)
        if valid is not None:
            self.valid = build_dataset_list(valid)

        # assert if either train/valid does not contain dataset..
        if ((self.train is None) ^ (self.valid is None)) or (
            isinstance(self.train, CombinedDataset)
            ^ isinstance(self.valid, CombinedDataset)
        ):
            raise RuntimeError("Both train and valid should be dataset class or None.")

        self.test_sets = {}
        if test is not None:
            for cfg in test:
                dataset = cfg.dataset
                if hasattr(cfg, "transform"):
                    transform = cfg.transform
                else:
                    transform = do_nothing_transform
                wrapped_transform = get_wrapped_transform(
                    is_espnet_preprocessor,
                    transform,
                    self.preprocessor,
                )
                self.test_sets[cfg.name] = DatasetWithTransform(
                    dataset, wrapped_transform, add_uid=is_espnet_preprocessor
                )

    @property
    def test(self):
        return self.test_sets


class DatasetWithTransform:
    """
    Lightweight wrapper for applying a transform function to dataset items.

    Args:
        dataset (Any): A dataset implementing __getitem__ and __len__.
        transform (Callable): A transform function applied to each sample.

    Example:
        >>> wrapped = DatasetWithTransform(my_dataset, my_transform)
        >>> item = wrapped[0]
    """

    def __init__(self, dataset, transform, add_uid=False):
        self.dataset = dataset
        self.transform = transform
        self.add_uid = add_uid

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.add_uid:
            return (str(idx), self.transform((str(idx), self.dataset[idx])))
        else:
            return self.transform(self.dataset[idx])

    def __call__(self, idx):
        return self.__getitem__(idx)
