"""Base system class and stage entrypoints for ESPnet3."""

import logging
from pathlib import Path

from omegaconf import DictConfig

from espnet3.systems.base.inference import infer
from espnet3.systems.base.metric import measure
from espnet3.systems.base.training import collect_stats, train

logger = logging.getLogger(__name__)


class BaseSystem:
    """Base class for all ESPnet3 systems.

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
        train_config (DictConfig | None): Training configuration.
        infer_config (DictConfig | None): Inference configuration.
        measure_config (DictConfig | None): Measurement configuration.
        stage_log_mapping (dict | None): Optional overrides for stage log path
            resolution. Keys are stage names; values are dotted attribute
            paths (e.g., ``"train_config.exp_dir"``) or lists/tuples of such
            paths (first non-empty value wins).

    Stage log mapping (base defaults):
        | Stage          | Path reference                     |
        |---             |---                                 |
        | create_dataset | train_config.recipe_dir            |
        | collect_stats  | train_config.stats_dir             |
        | train          | train_config.exp_dir               |
        | infer          | infer_config.inference_dir         |
        | measure        | measure_config.inference_dir       |
        | pack_model     | train_config.exp_dir               |
        | upload_model   | train_config.exp_dir               |

    Any stage missing from the mapping (or resolving to ``None``) falls back
    to the default log directory: ``train_config.exp_dir`` when available,
    otherwise ``<cwd>/logs``.

    Examples:
        Override a subset of stage log paths:
            ```python
            system = BaseSystem(
                train_config=train_cfg,
                infer_config=infer_cfg,
                measure_config=measure_cfg,
                stage_log_mapping={
                    "infer": "train_config.exp_dir",
                    "measure": "train_config.exp_dir",
                },
            )
            ```
    """

    def __init__(
        self,
        train_config: DictConfig | None = None,
        infer_config: DictConfig | None = None,
        measure_config: DictConfig | None = None,
        stage_log_mapping: dict | None = None,
    ) -> None:
        """Initialize the system with optional stage configs."""
        self.train_config = train_config
        self.infer_config = infer_config
        self.measure_config = measure_config

        if train_config is not None:
            self.exp_dir = Path(train_config.exp_dir)
            self.exp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.exp_dir = None

        if self.exp_dir is not None:
            default_dir = self.exp_dir
        else:
            default_dir = Path.cwd() / "logs"

        base_mapping = {
            "create_dataset": "train_config.recipe_dir",
            "collect_stats": "train_config.stats_dir",
            "train": "train_config.exp_dir",
            "infer": "infer_config.inference_dir",
            "measure": "measure_config.inference_dir",
            "pack_model": "train_config.exp_dir",
            "upload_model": "train_config.exp_dir",
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
            "Initialized %s with train_config=%s infer_config=%s "
            "measure_config=%s exp_dir=%s",
            self.__class__.__name__,
            train_config is not None,
            infer_config is not None,
            measure_config is not None,
            self.exp_dir,
        )

    def _resolve_stage_log_ref(
        self, ref: str | list[str] | tuple[str, ...] | None
    ) -> str | None:
        """Resolve stage log mapping references to concrete values.

        Supports dotted attribute paths like ``train_config.exp_dir`` and
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
        """Create datasets using the configured stage."""
        self._reject_stage_args("create_dataset", args, kwargs)
        logger.info("Running prepare() (BaseSystem stub). Nothing done.")

    def collect_stats(self, *args, **kwargs):
        """Collect statistics needed for training."""
        self._reject_stage_args("collect_stats", args, kwargs)
        logger.info(
            "Collecting stats | exp_dir=%s stats_dir=%s",
            getattr(self.train_config, "exp_dir", None),
            getattr(self.train_config, "stats_dir", None),
        )
        return collect_stats(self.train_config)

    def train(self, *args, **kwargs):
        """Train the system model."""
        self._reject_stage_args("train", args, kwargs)
        model_target = None
        if self.train_config is not None and hasattr(self.train_config, "model"):
            model_config = self.train_config.model
            if isinstance(model_config, DictConfig):
                model_target = model_config.get("_target_")
        logger.info(
            "Training start | exp_dir=%s model=%s",
            getattr(self.train_config, "exp_dir", None),
            model_target or "<unknown>",
        )
        return train(self.train_config)

    def infer(self, *args, **kwargs):
        """Run inference on the configured datasets."""
        self._reject_stage_args("infer", args, kwargs)
        logger.info(
            "Inference start | inference_dir=%s",
            getattr(self.infer_config, "inference_dir", None),
        )
        return infer(self.infer_config)

    def measure(self, *args, **kwargs):
        """Compute evaluation metrics from hypothesis/reference outputs."""
        self._reject_stage_args("measure", args, kwargs)
        logger.info(
            "Metrics start | measure_config=%s",
            self.measure_config is not None,
        )
        result = measure(self.measure_config)
        logger.info("results: %s", result)
        return result

    def publish(self, *args, **kwargs):
        """Publish artifacts from the experiment."""
        self._reject_stage_args("publish", args, kwargs)
        logger.info("Running publish() (BaseSystem stub). Nothing done.")
