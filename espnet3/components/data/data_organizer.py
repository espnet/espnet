"""DataOrganizer class for managing datasets in ESPnet3."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from hydra.utils import instantiate
from omegaconf import DictConfig

from espnet2.train.preprocessor import AbsPreprocessor
from espnet3.components.data.dataset import CombinedDataset, DatasetWithTransform


@dataclass
class DatasetConfig:
    """Configuration class for dataset metadata and construction.

    This class encapsulates the necessary fields to define and instantiate a dataset.
    Used with Hydra to allow modular and flexible configuration via YAML or
    dictionaries.

    Attributes:
        name (str): Name identifier for the dataset.
        path (Optional[str]): Optional path or ID required for dataset instantiation.
        dataset (Dict[str, Any]): A dictionary for Hydra instantiation of the dataset.
        transform (Optional[Dict[str, Any]]): A dictionary for Hydra instantiation of
            a transform applied to each sample after loading.

    Example:
        >>> config_dict = {
        ...     "name": "custom",
        ...     "dataset": {
        ...         "_target_": "my_project.datasets.MyDataset",
        ...     },
        ...     "transform": {
        ...         "_target_": "my_project.transforms.uppercase_transform"
        ...     }
        ... }
        >>> config = DatasetConfig(**config_dict)
    """

    name: str
    dataset: Dict[str, Any] = None
    transform: Optional[Dict[str, Any]] = None


def do_nothing(*x):
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


class DataOrganizer:
    """Organizes training, validation, and test datasets into a unified interface.

    This class constructs combined datasets for training and validation,
    and individual named datasets for testing, optionally applying a transform and
    preprocessor per dataset.

    Args:
        train (Optional[List[Union[DatasetConfig, Dict[str, Any]]]]):
            A list of training dataset configuration objects.
        valid (Optional[List[Union[DatasetConfig, Dict[str, Any]]]]):
            A list of validation dataset configuration objects.
        test (Optional[List[Union[DatasetConfig, Dict[str, Any]]]]):
            A list of test dataset configurations, each with a name and corresponding
            dataset and optional transform.
        preprocessor (Optional[Callable]): A global preprocessor function applied
            after each dataset's transform. If it's an instance of AbsPreprocessor,
            (uid, sample) is passed.

    Attributes:
        train (CombinedDataset): Combined dataset built from training configurations,
            or `None` if not provided.
        valid (CombinedDataset): Combined dataset built from validation configurations,
            or `None` if not provided.
        test_sets (Dict[str, DatasetWithTransform]): Dictionary mapping test set names
            to DatasetWithTransform instances.

    Raises:
        RuntimeError: If only one of `train` or `valid` is provided.
        RuntimeError: If `train` and `valid` are of mismatched types
            (e.g., one is CombinedDataset, the other is None).
        AssertionError: If `preprocessor` is not callable.

    Note:
        The `DataOrganizer` is designed to support both training and testing workflows:

        - For training: provide both `train` and `valid`.
        - For testing only: provide `test` and omit `train` / `valid`.
        - All three (`train`, `valid`, `test`) can also be provided simultaneously.

        If any of the `train`, `valid`, or `test` are omitted, the corresponding
            attributes will be set to `None` or empty.

    Example (training + validation):
        >>> organizer = DataOrganizer(
        ...     train=train_configs,
        ...     valid=valid_configs,
        ...     preprocessor=MyPreprocessor()
        ... )
        >>> sample = organizer.train[0]
        >>> test_sample = organizer.test["test_clean"][0]

    Example (testing only):
        >>> organizer = DataOrganizer(
        ...     test=test_configs,
        ...     preprocessor=MyPreprocessor()
        ... )
        >>> test_sample = organizer.test["test_clean"][0]
    """

    def __init__(
        self,
        train: Optional[List[Union[DatasetConfig, Dict[str, Any]]]] = None,
        valid: Optional[List[Union[DatasetConfig, Dict[str, Any]]]] = None,
        test: Optional[List[Union[DatasetConfig, Dict[str, Any]]]] = None,
        preprocessor: Optional[Callable[[dict], dict]] = None,
    ):
        """Initialize DataOrganizer object."""
        self.preprocessor = preprocessor or do_nothing
        if isinstance(self.preprocessor, (dict, DictConfig)):
            self.preprocessor = instantiate(self.preprocessor)
        assert callable(self.preprocessor), "Preprocessor should be callable."
        is_espnet_preprocessor = isinstance(self.preprocessor, AbsPreprocessor)

        def build_dataset_list(config_list):
            datasets = []
            transforms = []
            for config in config_list:
                if isinstance(config, dict):
                    config = DatasetConfig(**config)
                dataset = config.dataset
                if isinstance(dataset, (dict, DictConfig)):
                    dataset = instantiate(dataset)
                if hasattr(config, "transform"):
                    transform = config.transform
                else:
                    transform = do_nothing
                if isinstance(transform, (dict, DictConfig)):
                    transform = instantiate(transform)

                datasets.append(dataset)
                transforms.append((transform, self.preprocessor))
            return CombinedDataset(
                datasets,
                transforms,
                use_espnet_preprocessor=is_espnet_preprocessor,
            )

        self.train = None
        self.valid = None

        if train is not None:
            self.train = build_dataset_list(train)
        if valid is not None:
            self.valid = build_dataset_list(valid)

        # Check consistency between train and valid datasets:
        # - It is invalid to provide only one of train or valid.
        # - If one of them is a CombinedDataset, the other must be too.
        if (
            (
                (self.train is None and self.valid is not None)
                or (self.train is not None and self.valid is None)
            )
            or (
                isinstance(self.train, CombinedDataset)
                and not isinstance(self.valid, CombinedDataset)
            )
            or (
                not isinstance(self.train, CombinedDataset)
                and isinstance(self.valid, CombinedDataset)
            )
        ):
            raise RuntimeError("Both train and valid should be dataset class or None.")

        self.test_sets = {}
        if test is not None:
            for config in test:
                dataset = config.dataset
                if hasattr(config, "transform"):
                    transform = config.transform
                else:
                    transform = do_nothing
                self.test_sets[config.name] = DatasetWithTransform(
                    dataset,
                    transform,
                    self.preprocessor,
                    use_espnet_preprocessor=is_espnet_preprocessor,
                )

    @property
    def test(self):
        """Get the dictionary of test datasets."""
        return self.test_sets
