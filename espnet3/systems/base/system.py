"""Base system class and stage entrypoints for ESPnet3."""

import logging
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from espnet3.components.data.dataset_module import (
    load_dataset_module,
    parse_dataset_reference_config,
)
from espnet3.systems.base.inference import infer
from espnet3.systems.base.metric import measure
from espnet3.systems.base.training import collect_stats, train

logger = logging.getLogger(__name__)


class BaseSystem:
    """Base class for all ESPnet3 systems.

    Class Attributes:
        DATASET_BUILDER_CLASS_NAME: Name of the builder class expected in each
            dataset module (default ``"DatasetBuilder"``).
        DATASET_CLASS_NAME: Name of the dataset class expected in each dataset
            module (default ``"Dataset"``). Used by subclasses that instantiate
            datasets directly.

    Each system should implement the following:
      - create_dataset()
      - train()
      - infer()
      - measure()
      - publish()

    This class intentionally does NOT implement:
      - DAG
      - dependency checks
      - caching

    All behavior is config-driven.

    Args:
        training_config (DictConfig | None): Training configuration.
        inference_config (DictConfig | None): Inference configuration.
        metrics_config (DictConfig | None): Measurement configuration.
        stage_log_mapping (dict | None): Optional overrides for stage log path
            resolution. Keys are stage names; values are dotted attribute
            paths (e.g., ``"training_config.exp_dir"``) or lists/tuples of such
            paths (first non-empty value wins).

    Stage log mapping (base defaults):
        | Stage          | Path reference                     |
        |---             |---                                 |
        | create_dataset | training_config.data_dir           |
        | collect_stats  | training_config.stats_dir          |
        | train          | training_config.exp_dir            |
        | infer          | inference_config.inference_dir     |
        | measure        | metrics_config.inference_dir       |
        | pack_model     | training_config.exp_dir            |
        | upload_model   | training_config.exp_dir            |

    Any stage missing from the mapping (or resolving to ``None``) falls back
    to the default log directory: ``training_config.exp_dir`` when available,
    otherwise ``<cwd>/logs``.

    Examples:
        Override a subset of stage log paths:
            ```python
            system = BaseSystem(
                training_config=train_cfg,
                inference_config=infer_cfg,
                metrics_config=measure_cfg,
                stage_log_mapping={
                    "infer": "training_config.exp_dir",
                    "measure": "training_config.exp_dir",
                },
            )
            ```
    """

    DATASET_BUILDER_CLASS_NAME = "DatasetBuilder"
    DATASET_CLASS_NAME = "Dataset"

    def __init__(
        self,
        training_config: DictConfig | None = None,
        inference_config: DictConfig | None = None,
        metrics_config: DictConfig | None = None,
        stage_log_mapping: dict | None = None,
    ) -> None:
        """Initialize the system with optional stage configs."""
        self.training_config = training_config
        self.inference_config = inference_config
        self.metrics_config = metrics_config

        if training_config is not None:
            self.exp_dir = Path(training_config.exp_dir)
            self.exp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.exp_dir = None

        if self.exp_dir is not None:
            default_dir = self.exp_dir
        else:
            default_dir = Path.cwd() / "logs"

        base_mapping = {
            "create_dataset": "training_config.data_dir",
            "collect_stats": "training_config.stats_dir",
            "train": "training_config.exp_dir",
            "infer": "inference_config.inference_dir",
            "measure": "metrics_config.inference_dir",
            "pack_model": "training_config.exp_dir",
            "upload_model": "training_config.exp_dir",
        }
        mapping = dict(base_mapping)
        if stage_log_mapping:
            # Explicitly override base mapping with caller-provided values.
            mapping.update(stage_log_mapping)

        self.stage_log_dirs = {"default": default_dir}
        for stage, ref in mapping.items():
            resolved = self._resolve_stage_log_ref(ref)
            if resolved:
                self.stage_log_dirs[stage] = Path(resolved)

        logger.info(
            "Initialized %s with training_config=%s inference_config=%s "
            "metrics_config=%s exp_dir=%s",
            self.__class__.__name__,
            training_config is not None,
            inference_config is not None,
            metrics_config is not None,
            self.exp_dir,
        )

    def _resolve_stage_log_ref(
        self, ref: str | list[str] | tuple[str, ...] | None
    ) -> str | None:
        """Resolve stage log mapping references to concrete values.

        Supports dotted attribute paths like ``training_config.exp_dir`` and
        fallbacks via list/tuple entries (first non-empty value wins).
        """
        target = self
        if isinstance(ref, (list, tuple)):
            for item in ref:
                resolved = self._resolve_stage_log_ref(item)
                if resolved:
                    return resolved
            return None

        if not isinstance(ref, str):
            return None

        root_name, *parts = ref.split(".")
        current = getattr(target, root_name, None)
        for part in parts:
            if current is None:
                return None
            current = getattr(current, part, None)
        return current

    @staticmethod
    def _reject_stage_args(stage: str, args, kwargs) -> None:
        """Reject unexpected positional/keyword arguments for stages."""
        if args or kwargs:
            raise TypeError(
                f"Stage '{stage}' does not accept arguments. "
                "Put all settings in the YAML config."
            )

    # ---------------------------------------------------------
    # Stage stubs (override in subclasses if needed)
    # ---------------------------------------------------------
    def create_dataset(self, *args, **kwargs):
        """Create datasets from dataset references."""
        self._reject_stage_args("create_dataset", args, kwargs)
        logger.info(
            "%s.create_dataset(): starting dataset creation process",
            self.__class__.__name__,
        )
        start = time.perf_counter()
        dataset_config = getattr(self.training_config, "dataset", None)
        recipe_dir = getattr(self.training_config, "recipe_dir", None)
        create_dataset_config = getattr(
            self.training_config, "create_dataset", OmegaConf.create({})
        )
        default_builder_kwargs = dict(create_dataset_config)

        prepared_any = False

        if dataset_config is None:
            raise RuntimeError(
                "training_config.dataset must be set for create_dataset stage."
            )

        prepared_refs: set[str] = set()
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
                "training_config.dataset must include at least one entry in "
                "dataset.train / dataset.valid / dataset.test."
            )

        logger.info(
            "Dataset creation completed in %.2fs",
            time.perf_counter() - start,
        )
        return None

    def collect_stats(self, *args, **kwargs):
        """Collect statistics needed for training."""
        self._reject_stage_args("collect_stats", args, kwargs)
        logger.info(
            "Collecting stats | exp_dir=%s stats_dir=%s",
            getattr(self.training_config, "exp_dir", None),
            getattr(self.training_config, "stats_dir", None),
        )
        return collect_stats(self.training_config)

    def train(self, *args, **kwargs):
        """Train the system model."""
        self._reject_stage_args("train", args, kwargs)
        model_target = None
        if self.training_config is not None and hasattr(self.training_config, "model"):
            model_config = self.training_config.model
            if isinstance(model_config, DictConfig):
                model_target = model_config.get("_target_")
        logger.info(
            "Training start | exp_dir=%s model=%s",
            getattr(self.training_config, "exp_dir", None),
            model_target or "<unknown>",
        )
        return train(self.training_config)

    def infer(self, *args, **kwargs):
        """Run inference on the configured datasets."""
        self._reject_stage_args("infer", args, kwargs)
        logger.info(
            "Inference start | inference_dir=%s",
            getattr(self.inference_config, "inference_dir", None),
        )
        return infer(self.inference_config)

    def measure(self, *args, **kwargs):
        """Compute evaluation metrics from hypothesis/reference outputs."""
        self._reject_stage_args("measure", args, kwargs)
        logger.info(
            "Metrics start | metrics_config=%s",
            self.metrics_config is not None,
        )
        result = measure(self.metrics_config)
        logger.info("results: %s", result)
        return result

    def publish(self, *args, **kwargs):
        """Publish artifacts from the experiment."""
        self._reject_stage_args("publish", args, kwargs)
        logger.info("Running publish() (BaseSystem stub). Nothing done.")
