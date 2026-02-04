"""Base system class and stage entrypoints for ESPnet3."""

import logging
from pathlib import Path

from omegaconf import DictConfig

from espnet3.systems.base.inference import infer
from espnet3.systems.base.metric import metric
from espnet3.systems.base.training import collect_stats, train

logger = logging.getLogger(__name__)


class BaseSystem:
    """Base class for all ESPnet3 systems.

    Each system should implement the following:
      - create_dataset()
      - train()
      - infer()
      - metric()
      - publish()

    This class intentionally does NOT implement:
      - DAG
      - dependency checks
      - caching

    All behavior is config-driven.
    """

    def __init__(
        self,
        *,
        train_config: DictConfig | None = None,
        infer_config: DictConfig | None = None,
        metric_config: DictConfig | None = None,
    ) -> None:
        """Initialize the system with optional stage configs."""
        self.train_config = train_config
        self.infer_config = infer_config
        self.metric_config = metric_config
        if train_config is not None:
            self.exp_dir = Path(train_config.exp_dir)
            self.exp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.exp_dir = None
        logger.info(
            "Initialized %s with train_config=%s infer_config=%s "
            "metric_config=%s exp_dir=%s",
            self.__class__.__name__,
            train_config is not None,
            infer_config is not None,
            metric_config is not None,
            self.exp_dir,
        )

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

    def metric(self, *args, **kwargs):
        """Compute evaluation metrics from hypothesis/reference outputs."""
        self._reject_stage_args("metric", args, kwargs)
        logger.info(
            "metric start | metric_config=%s",
            self.metric_config is not None,
        )
        result = metric(self.metric_config)
        logger.info("metric results: %s", result)
        return result

    def publish(self, *args, **kwargs):
        """Publish artifacts from the experiment."""
        self._reject_stage_args("publish", args, kwargs)
        logger.info("Running publish() (BaseSystem stub). Nothing done.")
