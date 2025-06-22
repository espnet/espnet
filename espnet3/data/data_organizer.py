from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from espnet2.train.preprocessor import AbsPreprocessor
from espnet3.data.dataset import CombinedDataset, DatasetWithTransform


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
        assert callable(self.preprocessor), \
            "Preprocessor should be callable."
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

                datasets.append(dataset)
                transforms.append((transform, self.preprocessor))
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
                self.test_sets[cfg.name] = DatasetWithTransform(
                    dataset,
                    transform,
                    self.preprocessor,
                    add_uid=is_espnet_preprocessor,
                )

    @property
    def test(self):
        return self.test_sets
