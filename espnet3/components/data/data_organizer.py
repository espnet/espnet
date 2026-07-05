"""DataOrganizer class for managing datasets in ESPnet3."""

import copy
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
        preprocessor (Optional[Callable]): A global preprocessor function or Hydra
            config applied after each dataset's transform. If it is an instance of
            ``AbsPreprocessor``, each sample is passed as ``(uid, sample)``.
            Dict-style configs support both shared and split-specific forms:

            - Shared config: the same preprocessor is used for train/valid/test.
            - Split config: ``train``, ``valid``, and ``test`` can each define their
              own ``_target_`` block.
            - Shared keys outside ``train``/``valid``/``test`` are merged into each
              split-specific config. This is useful for common tokenizer settings
              such as ``token_type``, ``token_list``, and ``bpemodel``.

            Split config fallback rules are:

            - If ``test`` is missing, test uses the valid preprocessor.
            - If ``valid`` is missing, valid uses the train preprocessor.
            - If ``train`` is missing, train uses no preprocessor.
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

    Notes:
        The `DataOrganizer` is designed to support both training and testing
        workflows:

        - For training: provide both `train` and `valid`.
        - For testing only: provide `test` and omit `train` / `valid`.
        - All three (`train`, `valid`, `test`) can also be provided simultaneously.

        If any of the `train`, `valid`, or `test` are omitted, the corresponding
            attributes will be set to `None` or empty.

        Split-specific preprocessor configs are useful when train and valid should
        not share augmentation behavior.

    Examples:
        Training and validation with one shared preprocessor:
        >>> organizer = DataOrganizer(
        ...     train=training_configs,
        ...     valid=valid_configs,
        ...     preprocessor=MyPreprocessor()
        ... )
        >>> sample = organizer.train[0]

        Testing only:
        >>> organizer = DataOrganizer(
        ...     test=test_configs,
        ...     preprocessor=MyPreprocessor()
        ... )
        >>> test_sample = organizer.test["test_clean"][0]

        Shared Hydra config for all splits:
            >>> config = {
            ...     "_target_": "my_project.preprocess.BuildPreprocessor",
            ...     "token_type": "bpe",
            ...     "token_list": "tokens.txt",
            ... }
            >>> organizer = DataOrganizer(
            ...     train=training_configs,
            ...     valid=valid_configs,
            ...     preprocessor=config,
            ... )

        Split-specific Hydra config with shared tokenizer settings:
            >>> config = {
            ...     "token_type": "bpe",
            ...     "token_list": "tokens.txt",
            ...     "bpemodel": "bpe.model",
            ...     "train": {
            ...         "_target_": "my_project.preprocess.TrainPreprocessor",
            ...         "speed_perturb_prob": 0.1,
            ...     },
            ...     "valid": {
            ...         "_target_": "my_project.preprocess.ValidPreprocessor",
            ...     },
            ... }
            >>> organizer = DataOrganizer(
            ...     train=training_configs,
            ...     valid=valid_configs,
            ...     test=test_configs,
            ...     preprocessor=config,
            ... )

        Split-specific config with test fallback to valid:
            >>> config = {
            ...     "train": {
            ...         "_target_": "my_project.preprocess.TrainPreprocessor",
            ...     },
            ...     "valid": {
            ...         "_target_": "my_project.preprocess.ValidPreprocessor",
            ...     },
            ... }
            >>> organizer = DataOrganizer(
            ...     train=training_configs,
            ...     valid=valid_configs,
            ...     test=test_configs,
            ...     preprocessor=config,
            ... )
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
        self.recipe_dir = recipe_dir

        preprocessor_cfg = preprocessor
        if preprocessor_cfg is None:
            is_espnet_preprocessor = False
            train_preprocessor = do_nothing
            valid_preprocessor = do_nothing
            test_preprocessor = do_nothing
            train_use_espnet_preprocessor = False
            valid_use_espnet_preprocessor = False
            test_use_espnet_preprocessor = False
        elif isinstance(preprocessor_cfg, (dict, DictConfig)):
            plain_cfg = (
                OmegaConf.to_container(preprocessor_cfg, resolve=True)
                if isinstance(preprocessor_cfg, DictConfig)
                else dict(preprocessor_cfg)
            )
            has_split_configs = any(
                isinstance(plain_cfg.get(split), dict)
                for split in ("train", "valid", "test")
            )
            if has_split_configs:
                shared_cfg = {
                    key: value
                    for key, value in plain_cfg.items()
                    if key not in ("train", "valid", "test")
                }
                train_cfg = plain_cfg.get("train")
                valid_cfg = plain_cfg.get("valid")
                test_cfg = plain_cfg.get("test")
                if valid_cfg is None:
                    logger.warning(
                        "Split preprocessor config is missing 'valid'. "
                        "The valid split will use the train preprocessor."
                    )
                    valid_cfg = train_cfg
                if train_cfg is None:
                    logger.warning(
                        "Split preprocessor config is missing 'train'. "
                        "The train split will not use a preprocessor."
                    )
                if test_cfg is None:
                    logger.warning(
                        "Split preprocessor config is missing 'test'. "
                        "The test split will use the valid preprocessor."
                    )
                    test_cfg = valid_cfg
                if train_cfg is not None and shared_cfg:
                    train_cfg = OmegaConf.merge(shared_cfg, train_cfg)
                elif train_cfg is None:
                    train_cfg = shared_cfg or None
                if valid_cfg is not None and shared_cfg:
                    valid_cfg = OmegaConf.merge(shared_cfg, valid_cfg)
                elif valid_cfg is None:
                    valid_cfg = shared_cfg or None
                if test_cfg is not None and shared_cfg and test_cfg is not valid_cfg:
                    test_cfg = OmegaConf.merge(shared_cfg, test_cfg)
                train_preprocessor, train_is_espnet = (
                    self._instantiate_preprocessor_from_config(train_cfg, True)
                )
                valid_preprocessor, valid_is_espnet = (
                    self._instantiate_preprocessor_from_config(valid_cfg, False)
                )
                test_preprocessor, test_is_espnet = (
                    self._instantiate_preprocessor_from_config(test_cfg, False)
                )
                train_use_espnet_preprocessor = train_is_espnet is True
                valid_use_espnet_preprocessor = valid_is_espnet is True
                test_use_espnet_preprocessor = test_is_espnet is True
                is_espnet_preprocessor = any(
                    (
                        train_use_espnet_preprocessor,
                        valid_use_espnet_preprocessor,
                        test_use_espnet_preprocessor,
                    )
                )
                train_preprocessor = train_preprocessor or do_nothing
                valid_preprocessor = valid_preprocessor or do_nothing
                test_preprocessor = test_preprocessor or do_nothing
            else:
                # Use _partial_=True to obtain the target class without calling
                # __init__, so a missing 'train' argument does not raise here.
                partial_preprocessor = instantiate(preprocessor_cfg, _partial_=True)
                is_espnet_preprocessor = (
                    hasattr(partial_preprocessor, "func")
                    and isinstance(partial_preprocessor.func, type)
                    and issubclass(partial_preprocessor.func, AbsPreprocessor)
                )
                if is_espnet_preprocessor:
                    if "train" in preprocessor_cfg:
                        logger.warning(
                            "Preprocessor config contains a 'train' field, but "
                            "DataOrganizer sets it automatically (True for train "
                            "split, False for valid/test). The config value will "
                            "be ignored."
                        )
                    train_preprocessor = instantiate(
                        OmegaConf.merge(preprocessor_cfg, {"train": True})
                    )
                    valid_preprocessor = instantiate(
                        OmegaConf.merge(preprocessor_cfg, {"train": False})
                    )
                    train_use_espnet_preprocessor = True
                    valid_use_espnet_preprocessor = True
                else:
                    train_preprocessor = instantiate(preprocessor_cfg)
                    valid_preprocessor = train_preprocessor
                    train_use_espnet_preprocessor = False
                    valid_use_espnet_preprocessor = False
                test_use_espnet_preprocessor = valid_use_espnet_preprocessor
                test_preprocessor = valid_preprocessor
        elif isinstance(preprocessor_cfg, AbsPreprocessor):
            is_espnet_preprocessor = True
            # Already-instantiated AbsPreprocessor: deepcopy for the train
            # split; keep the original as valid/test so callers that inspect the
            # instance (e.g. test-only setups) see side-effects.
            if train is not None:
                train_preprocessor = copy.deepcopy(preprocessor_cfg)
                train_preprocessor.train = True
            else:
                train_preprocessor = preprocessor_cfg
            valid_preprocessor = preprocessor_cfg
            valid_preprocessor.train = False
            test_preprocessor = valid_preprocessor
            train_use_espnet_preprocessor = True
            valid_use_espnet_preprocessor = True
            test_use_espnet_preprocessor = True
        else:
            # Already-instantiated custom callable (non-AbsPreprocessor).
            assert callable(preprocessor_cfg), "Preprocessor should be callable."
            is_espnet_preprocessor = False
            train_preprocessor = preprocessor_cfg
            valid_preprocessor = preprocessor_cfg
            test_preprocessor = preprocessor_cfg
            train_use_espnet_preprocessor = False
            valid_use_espnet_preprocessor = False
            test_use_espnet_preprocessor = False

        assert callable(train_preprocessor), "Preprocessor should be callable."
        self.preprocessor = train_preprocessor

        self.train = None
        self.valid = None

        if train is not None:
            self.train = self._build_dataset_list(
                train,
                train_preprocessor,
                train_use_espnet_preprocessor,
            )
        if valid is not None:
            self.valid = self._build_dataset_list(
                valid,
                valid_preprocessor,
                valid_use_espnet_preprocessor,
            )

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
                    test_preprocessor,
                    use_espnet_preprocessor=test_use_espnet_preprocessor,
                )

    def _instantiate_preprocessor_from_config(
        self,
        preprocessor_cfg: Dict[str, Any] | DictConfig | None,
        train_flag: bool,
    ):
        """Instantiate one preprocessor config for the requested split."""
        if preprocessor_cfg is None:
            return None, None
        partial_preprocessor = instantiate(preprocessor_cfg, _partial_=True)
        is_espnet = (
            hasattr(partial_preprocessor, "func")
            and isinstance(partial_preprocessor.func, type)
            and issubclass(partial_preprocessor.func, AbsPreprocessor)
        )
        if is_espnet:
            if "train" in preprocessor_cfg:
                logger.warning(
                    "Preprocessor config contains a 'train' field, but "
                    "DataOrganizer sets it automatically (True for train "
                    "split, False for valid/test). The config value will "
                    "be ignored."
                )
            preprocessor_cfg = OmegaConf.merge(
                preprocessor_cfg, {"train": train_flag}
            )
        return instantiate(preprocessor_cfg), is_espnet

    def _build_dataset_list(
        self,
        config_list,
        preprocessor,
        use_espnet_preprocessor: bool,
    ):
        """Build a combined dataset from config entries."""
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
            transforms.append((transform, preprocessor))

        return CombinedDataset(
            datasets,
            transforms,
            use_espnet_preprocessor=use_espnet_preprocessor,
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
