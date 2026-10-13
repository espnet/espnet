"""Base system class and stage entrypoints for ESPnet3."""

import logging
from pathlib import Path
from typing import ClassVar, Mapping, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

from espnet3.components.data.dataset_module import (
    load_dataset_module,
    parse_dataset_reference_config,
)
from espnet3.contract.stages import StageSpec, check_stage_contract, config_roles
from espnet3.publication.demo.packing import pack_demo as _pack_demo
from espnet3.publication.demo.packing import upload_demo as _upload_demo
from espnet3.systems.base.inference import infer
from espnet3.systems.base.metric import measure
from espnet3.systems.base.training import collect_stats, train
from espnet3.utils.publication_utils import pack_model as _pack_model
from espnet3.utils.publication_utils import upload_model as _upload_model
from espnet3.utils.run_utils import ConfigError
from espnet3.utils.stage_configs import build_stage_configs

logger = logging.getLogger(__name__)


def _exp_dir_from(
    system_cls: type, configs: Mapping[str, Optional[DictConfig]]
) -> Optional[Path]:
    """Return the first given config's own, self-resolving `exp_dir`.

    Walks `system_cls.stages` in order, by role (each role considered
    once); for the first role whose own config is given and whose
    `exp_dir` resolves without depending on anything outside that one
    config, returns it. A config that cannot resolve on its own (e.g. it
    depends on `exp_tag` from a different role) is skipped, not an error.

    Args:
        system_cls: The system class whose declared `stages` sets the
            role order.
        configs: Each config role's own config, as given to `__init__`.

    Returns:
        The first resolvable `exp_dir`, or `None` if no given config has
        one.
    """
    seen_roles = set()
    for spec in system_cls.stages:
        for role in config_roles(spec.config):
            if role in seen_roles:
                continue
            seen_roles.add(role)
            config = configs.get(role)
            if config is None:
                continue
            try:
                standalone = OmegaConf.create(
                    OmegaConf.to_container(config, resolve=False)
                )
                OmegaConf.resolve(standalone)
            except Exception:
                continue
            value = standalone.get("exp_dir")
            if value:
                return Path(value)
    return None


class BaseSystem:
    """Base class for all ESPnet3 systems.

    A system declares its stages as the class attribute ``stages``: a
    tuple of :class:`~espnet3.contract.stages.StageSpec`, in
    the order "all" runs them. Every public method a system defines
    (class-defined, not starting with ``_``) must be one of these stage
    names - a system has no other public surface. See
    ``espnet3.contract.stages.check_stage_contract`` for the
    checks run when a subclass is defined, and
    ``espnet3.contract.stages.stage_log_dir`` for how a
    stage's log directory is resolved.

    Class Attributes:
        DATASET_BUILDER_CLASS_NAME: Name of the builder class expected in each
            dataset module (default ``"DatasetBuilder"``).
        DATASET_CLASS_NAME: Name of the dataset class expected in each dataset
            module (default ``"Dataset"``). Used by subclasses that instantiate
            datasets directly.
        stages: The declared stage order (see above).

    All behavior is config-driven.

    Examples:
        >>> system = BaseSystem(
        ...     configs={"training": OmegaConf.create({"exp_dir": "./exp/x"})}
        ... )
        >>> [s.name for s in system.stages[:3]]
        ['create_dataset', 'collect_stats', 'train']
    """

    DATASET_BUILDER_CLASS_NAME = "DatasetBuilder"
    DATASET_CLASS_NAME = "Dataset"

    stages: ClassVar[Tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(name="collect_stats", config="training", log_dir="stats_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
        StageSpec(name="pack_model", config=("training", "publication")),
        StageSpec(name="upload_model", config="publication"),
        StageSpec(name="pack_demo", config="demo", log_dir="pack.out_dir"),
        StageSpec(name="upload_demo", config="demo", log_dir="pack.out_dir"),
    )

    def __init_subclass__(cls, **kwargs) -> None:
        """Check the subclass's declared ``stages`` against its own methods."""
        super().__init_subclass__(**kwargs)
        check_stage_contract(cls)

    def __init__(
        self,
        *,
        configs: Mapping[str, Optional[DictConfig]],
        exp_dir: Optional[Path] = None,
        stages_to_run: Optional[Sequence[str]] = None,
    ) -> None:
        """Build each requested stage's config and hold it, by stage name.

        Args:
            configs: Each config role's own config (its default plus
                whatever the caller loaded), not yet resolved, keyed by
                role name (``"training"``, ``"inference"``, ...). A role
                this system has no stage for is ignored; a role it does
                have a stage for but that is missing here is treated as
                empty. A stage that needs its own ``recipe_dir`` reads it
                from its own config (``self.stage_configs[stage].recipe_dir``),
                not from a separate constructor argument.
            exp_dir: The experiment directory to bake stage configs under
                and read earlier baked configs from. When `None`, derived
                from the first given config whose own `exp_dir` resolves
                by itself (see `_exp_dir_from`); if none does, raises.
            stages_to_run: The stages this run actually requests. Only
                these (and the earlier stages they inherit from) have
                their config built and resolved; a later stage's missing
                identity never stops construction. `None` builds every
                declared stage.

        Raises:
            ConfigError: `exp_dir` is not given and no config in
                `configs` has a self-resolving `exp_dir`.

        Notes:
            Each stage's config is its own `DictConfig` object - the
            earlier stages' own keys (this run's, in memory, or an
            earlier run's baked file), then this stage's own config on
            top, then resolved. No two stages share one config object, so
            one stage popping or overwriting a key never affects another.
        """
        resolved_exp_dir = (
            Path(exp_dir) if exp_dir is not None else _exp_dir_from(type(self), configs)
        )
        if resolved_exp_dir is None:
            raise ConfigError(
                f"{type(self).__name__} needs an experiment directory; pass "
                "--exp_dir, or set exp_dir in one of the given configs."
            )
        self.exp_dir = resolved_exp_dir
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self._default_log_dir = self.exp_dir

        self.stage_configs, self.own_keys = build_stage_configs(
            type(self), configs, exp_dir=self.exp_dir, upto=stages_to_run
        )

        logger.info(
            "Initialized %s exp_dir=%s stages=%s",
            self.__class__.__name__,
            self.exp_dir,
            sorted(self.stage_configs),
        )

    @staticmethod
    def _reject_stage_args(stage: str, args, kwargs) -> None:
        """Reject unexpected positional/keyword arguments for stages."""
        if args or kwargs:
            raise TypeError(
                f"Stage '{stage}' does not accept arguments. "
                "Put all settings in the YAML config."
            )

    @staticmethod
    def _require(config, key: str, message: str):
        """Return ``config[key]``, raising ``RuntimeError`` when missing.

        Args:
            config: Dict-like config (e.g. ``DictConfig``) to read from.
                ``None`` is treated the same as a missing key.
            key: Field name to extract.
            message: Message for the ``RuntimeError`` raised when the
                field is absent or ``None``. By convention, names the
                stage and the key (e.g. ``"infer: the stage config has
                no inference_dir; ..."``), not the config role.

        Returns:
            The value stored under ``key``.

        Raises:
            RuntimeError: If ``config`` is ``None`` or ``config[key]`` is
                missing or ``None``.

        Examples:
            >>> from omegaconf import OmegaConf
            >>> BaseSystem._require(OmegaConf.create({"a": 1}), "a", "need a")
            1
        """
        value = config.get(key, None) if config is not None else None
        if value is None:
            raise RuntimeError(message)
        return value

    # ---------------------------------------------------------
    # Stage stubs (override in subclasses if needed)
    # ---------------------------------------------------------
    def create_dataset(self, *args, **kwargs):
        """Create datasets from dataset references.

        Examples:
            >>> system = BaseSystem(configs={"training": training_config})
            >>> system.create_dataset()
        """
        self._reject_stage_args("create_dataset", args, kwargs)
        config = self.stage_configs["create_dataset"]
        logger.info(
            "%s.create_dataset(): starting dataset creation process",
            self.__class__.__name__,
        )
        dataset_config = getattr(config, "dataset", None)
        recipe_dir = getattr(config, "recipe_dir", None)
        create_dataset_config = getattr(config, "create_dataset", OmegaConf.create({}))
        default_builder_kwargs = dict(create_dataset_config)

        prepared_any = False

        if dataset_config is None:
            raise RuntimeError(
                "create_dataset: the stage config has no dataset; set "
                "dataset: in training.yaml."
            )

        prepared_refs: set = set()
        for split_name in ("train", "valid", "test"):
            entries = getattr(dataset_config, split_name, None)
            if entries is None:
                continue
            for entry in entries:
                plain = dict(entry)
                data_src, _ = parse_dataset_reference_config(plain)
                if data_src in prepared_refs:
                    continue
                prepared_refs.add(data_src)

                builder_kwargs = dict(default_builder_kwargs)

                module = load_dataset_module(data_src=data_src, recipe_dir=recipe_dir)
                builder = getattr(module, self.DATASET_BUILDER_CLASS_NAME)()
                logger.info("Ensuring dataset is prepared: %s", data_src or "local")

                # Ensure raw source exists first, then build task-ready artifacts.
                if not builder.is_source_prepared(**builder_kwargs):
                    builder.prepare_source(**builder_kwargs)

                if not builder.is_built(**builder_kwargs):
                    builder.build(**builder_kwargs)
                prepared_any = True

        if not prepared_any:
            raise RuntimeError(
                "create_dataset: the stage config's dataset has no entry in "
                "dataset.train / dataset.valid / dataset.test."
            )

        logger.info("Dataset creation completed.")
        return None

    def collect_stats(self, *args, **kwargs):
        """Collect statistics needed for training.

        Reads the same ``dataset:`` block as ``train``. By default the
        ``DataOrganizer`` is built with ``train_mode=False``, so the train
        preprocessor's augmentation (``data_aug_effects``, speed perturbation
        and so on) is not applied, as in ESPnet2: the statistics and the shape
        files the sampler batches by are then the same on every run. ``train``
        still augments. To collect statistics with the augmentation applied,
        set it in the training config:

        .. code-block:: yaml

            collect_stats_train_mode: true

        Examples:
            ```python
            system = ASRSystem(
                configs={"training": training_config}, exp_dir=exp_dir
            )
            system.collect_stats()
            # Writes stats under the stage config's stats_dir.
            ```
        """
        self._reject_stage_args("collect_stats", args, kwargs)
        config = self.stage_configs["collect_stats"]
        logger.info(
            "Collecting stats | exp_dir=%s stats_dir=%s",
            getattr(config, "exp_dir", None),
            getattr(config, "stats_dir", None),
        )
        return collect_stats(config)

    def train(self, *args, **kwargs):
        """Train the system model.

        Examples:
            ```python
            system = ASRSystem(
                configs={"training": training_config}, exp_dir=exp_dir
            )
            system.train()
            # Runs model.fit() and saves the resolved config under exp_dir.
            ```
        """
        self._reject_stage_args("train", args, kwargs)
        config = self.stage_configs["train"]
        model_target = None
        if hasattr(config, "model") and isinstance(config.model, DictConfig):
            model_target = config.model.get("_target_")
        logger.info(
            "Training start | exp_dir=%s model=%s",
            getattr(config, "exp_dir", None),
            model_target or "<unknown>",
        )
        return train(config)

    def infer(self, *args, **kwargs):
        """Run inference on the configured datasets.

        Examples:
            ```python
            system = ASRSystem(
                configs={"inference": inference_config}, exp_dir=exp_dir
            )
            system.infer()
            # Writes decoded hypotheses under the stage config's inference_dir.
            ```
        """
        self._reject_stage_args("infer", args, kwargs)
        config = self.stage_configs["infer"]
        self._require(
            config,
            "inference_dir",
            "infer: the stage config has no inference_dir; set it in "
            "inference.yaml or pass --exp_dir of a run that trained.",
        )
        logger.info("Inference start | inference_dir=%s", config.inference_dir)
        return infer(config)

    def measure(self, *args, **kwargs):
        """Compute evaluation metrics from hypothesis/reference outputs.

        Examples:
            ```python
            system = ASRSystem(
                configs={"metrics": metrics_config}, exp_dir=exp_dir
            )
            result = system.measure()
            # result holds the computed metric values.
            ```
        """
        self._reject_stage_args("measure", args, kwargs)
        config = self.stage_configs["measure"]
        self._require(
            config,
            "inference_dir",
            "measure: the stage config has no inference_dir; pass "
            "--inference_config, or --exp_dir of the run that ran infer.",
        )
        logger.info("Metrics start | inference_dir=%s", config.inference_dir)
        result = measure(config, inference_config=self.stage_configs["infer"])
        logger.info("results: %s", result)
        return result

    # ---------------------------------------------------------
    # Publication stages (optional overrides)
    # ---------------------------------------------------------
    def pack_model(self, *args, **kwargs):
        """Pack model artifacts into an espnet3 bundle.

        Examples:
            ```python
            system = ASRSystem(
                configs={
                    "training": training_config,
                    "publication": publication_config,
                },
                exp_dir=exp_dir,
            )
            system.pack_model()
            # Packs model artifacts under the stage config's pack_model.out_dir.
            ```
        """
        self._reject_stage_args("pack_model", args, kwargs)
        return _pack_model(
            training_config=self.stage_configs.get("train"),
            publication_config=self.stage_configs.get("pack_model"),
            inference_config=self.stage_configs.get("infer"),
            metrics_config=self.stage_configs.get("measure"),
        )

    def upload_model(self, *args, **kwargs):
        """Upload model bundle to HuggingFace."""
        self._reject_stage_args("upload_model", args, kwargs)
        return _upload_model(self)

    def pack_demo(self, *args, **kwargs):
        """Pack demo assets into a runnable demo directory."""
        self._reject_stage_args("pack_demo", args, kwargs)
        return _pack_demo(self)

    def upload_demo(self, *args, **kwargs):
        """Upload demo bundle to HuggingFace."""
        self._reject_stage_args("upload_demo", args, kwargs)
        return _upload_demo(self)


# __init_subclass__ only fires for subclasses; check BaseSystem's own
# declaration here so it is held to the same contract.
check_stage_contract(BaseSystem)
