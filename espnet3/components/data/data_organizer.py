"""DataOrganizer class for managing datasets in ESPnet3."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet2.train.preprocessor import AbsPreprocessor
from espnet3.components.data.dataset import (
    CombinedDataset,
    DatasetWithTransform,
    do_nothing,
)
from espnet3.components.data.dataset_module import instantiate_dataset_reference
from espnet3.utils.logging_utils import build_callable_name, build_qualified_name

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration class for dataset metadata and construction.

    This class encapsulates the necessary fields to define and instantiate a dataset.
    Used with Hydra to allow modular and flexible configuration via YAML or
    dictionaries.

    Attributes:
        name (str): Name identifier for the dataset.
        data_src (Optional[str]): Optional dataset source reference resolved via
            ``espnet3.components.data.dataset_module``.
        data_src_args (Optional[Dict[str, Any]]): Keyword arguments passed to the
            recipe ``Dataset`` class.
        transform (Optional[Dict[str, Any]]): A dictionary for Hydra instantiation of
            a transform applied to each sample after loading.
        split (Optional[str]): Optional split label kept for compatibility with
            configs that still carry it as metadata.

    Examples:
        Recipe-backed dataset entry:
            >>> config_dict = {
            ...     "name": "custom",
            ...     "data_src": "mini_an4/asr",
            ...     "data_src_args": {"split": "test"},
            ...     "transform": {
            ...         "_target_": "my_project.transforms.uppercase_transform",
            ...     },
            ... }
            >>> config = DatasetConfig(**config_dict)
            >>> config.data_src
            'mini_an4/asr'

        Local recipe dataset entry:
            >>> config = DatasetConfig(
            ...     name="local_eval",
            ...     data_src_args={"split": "eval"},
            ... )
            >>> config.data_src is None
            True
    """

    name: Optional[str] = None
    data_src: Optional[str] = None
    data_src_args: Optional[Dict[str, Any]] = None
    transform: Optional[Dict[str, Any]] = None
    split: Optional[str] = None


def _log_dataset(
    log: logging.Logger,
    label: str,
    combined: CombinedDataset | DatasetWithTransform | None,
) -> None:
    """Log dataset structure and transforms for a given label."""
    if combined is None:
        log.info("%s dataset: None", label)
        return

    log.info("%s dataset: %s", label, build_qualified_name(combined))
    if isinstance(combined, DatasetWithTransform):
        log.info(
            "%s dataset detail: %s(len=%s) transform=%s preprocessor=%s",
            label,
            build_qualified_name(combined.dataset),
            len(combined),
            build_callable_name(combined.transform),
            build_callable_name(combined.preprocessor),
        )
        return

    for idx, (dataset, (transform, preprocessor)) in enumerate(
        zip(combined.datasets, combined.transforms)
    ):
        log.info(
            "%s dataset[%d]: %s(len=%s) transform=%s preprocessor=%s",
            label,
            idx,
            build_qualified_name(dataset),
            len(dataset),
            build_callable_name(transform),
            build_callable_name(preprocessor),
        )


class DataOrganizer:
    """Organizes training, validation, and test datasets into a unified interface.

    This class constructs combined datasets for training and validation,
    and individual named datasets for testing, optionally applying a transform and
    preprocessor per dataset.

    Args:
        train (Optional[List[Union[DatasetConfig, Dict[str, Any], DictConfig]]]):
            A list of training dataset configuration objects.
        valid (Optional[List[Union[DatasetConfig, Dict[str, Any], DictConfig]]]):
            A list of validation dataset configuration objects.
        test (Optional[List[Union[DatasetConfig, Dict[str, Any], DictConfig]]]):
            A list of test dataset configurations, each with a name and corresponding
            data source and optional transform.
        preprocessor (Optional[Callable]): A global preprocessor function applied
            after each dataset's transform. If it's an instance of AbsPreprocessor,
            (uid, sample) is passed.
        recipe_dir (Optional[str]): Recipe root used to resolve local dataset modules
            when dataset entries omit ``data_src`` and rely on
            ``recipe_dir/dataset``.

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
        ...     train=training_configs,
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
        train: Optional[List[Union[DatasetConfig, Dict[str, Any], DictConfig]]] = None,
        valid: Optional[List[Union[DatasetConfig, Dict[str, Any], DictConfig]]] = None,
        test: Optional[List[Union[DatasetConfig, Dict[str, Any], DictConfig]]] = None,
        preprocessor: Optional[Callable[[dict], dict]] = None,
        recipe_dir: Optional[str] = None,
    ):
        """Initialize DataOrganizer object."""
        self.preprocessor = preprocessor or do_nothing
        self.recipe_dir = recipe_dir
        if isinstance(self.preprocessor, (dict, DictConfig)):
            self.preprocessor = instantiate(self.preprocessor)
        assert callable(self.preprocessor), "Preprocessor should be callable."
        is_espnet_preprocessor = isinstance(self.preprocessor, AbsPreprocessor)

        def build_dataset_list(config_list):
            datasets = []
            transforms = []
            for config in config_list:
                if isinstance(config, DictConfig):
                    raw_config = OmegaConf.to_container(config, resolve=True)
                else:
                    raw_config = config
                dataset = instantiate_dataset_reference(
                    raw_config,
                    recipe_dir=self.recipe_dir,
                )

                transform = raw_config.get("transform", do_nothing)

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
                if isinstance(config, DictConfig):
                    raw_config = OmegaConf.to_container(config, resolve=True)
                else:
                    raw_config = config
                dataset = instantiate_dataset_reference(
                    raw_config,
                    recipe_dir=self.recipe_dir,
                )

                transform = raw_config.get("transform", do_nothing)

                if isinstance(transform, (dict, DictConfig)):
                    transform = instantiate(transform)

                name = raw_config.get("name") or str(
                    raw_config.get("data_src") or "local"
                )
                self.test_sets[name] = DatasetWithTransform(
                    dataset,
                    transform,
                    self.preprocessor,
                    use_espnet_preprocessor=is_espnet_preprocessor,
                )

    @property
    def test(self):
        """Get the dictionary of test datasets."""
        return self.test_sets

    def __repr__(self) -> str:
        """Return a compact, human-readable representation for debugging."""
        train_desc = (
            f"{build_qualified_name(self.train)}(len={len(self.train)})"
            if self.train is not None
            else "None"
        )
        valid_desc = (
            f"{build_qualified_name(self.valid)}(len={len(self.valid)})"
            if self.valid is not None
            else "None"
        )
        test_entries = []
        for name, dataset in self.test_sets.items():
            test_entries.append(
                f"{name}: {build_qualified_name(dataset)}(len={len(dataset)})"
            )
        tests_desc = ", ".join(test_entries) if test_entries else "None"
        return (
            f"{self.__class__.__name__}("
            f"preprocessor={build_callable_name(self.preprocessor)}, "
            f"train={train_desc}, "
            f"valid={valid_desc}, "
            f"test={tests_desc}"
            f")"
        )

    def log_summary(self, log: logging.Logger | None = None) -> None:
        """Log a concise dataset summary."""
        log = log or logger
        log.info("Data organizer: %s", build_qualified_name(self))
        _log_dataset(log, "train", self.train)
        _log_dataset(log, "valid", self.valid)

        if not self.test_sets:
            log.info("test datasets: None")
            return

        log.info("test datasets: %d", len(self.test_sets))
        for name, dataset in self.test_sets.items():
            _log_dataset(log, f"test[{name}]", dataset)
