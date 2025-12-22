# system_base.py

import logging
from pathlib import Path

from omegaconf import DictConfig

from espnet3.systems.base.inference import inference
from espnet3.systems.base.score import score
from espnet3.systems.base.train import collect_stats, train

logger = logging.getLogger(__name__)


class BaseSystem:
    """Base class for all ESPnet3 systems.

    Each system should implement the following:
      - create_dataset()
      - train()
      - decode()
      - score()
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
        measure_config: DictConfig | None = None,
    ) -> None:
        self.train_config = train_config
        self.infer_config = infer_config
        self.metric_config = measure_config
        if train_config is not None:
            self.exp_dir = Path(train_config.exp_dir)
            self.exp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.exp_dir = None
        logger.info(
            "Initialized %s with train_config=%s infer_config=%s "
            "measure_config=%s exp_dir=%s",
            self.__class__.__name__,
            train_config is not None,
            infer_config is not None,
            measure_config is not None,
            self.exp_dir,
        )

    @staticmethod
    def _reject_stage_args(stage: str, args, kwargs) -> None:
        if args or kwargs:
            raise TypeError(
                f"Stage '{stage}' does not accept arguments. "
                "Put all settings in the YAML config."
            )

    # ---------------------------------------------------------
    # Stage stubs (override in subclasses if needed)
    # ---------------------------------------------------------
    def create_dataset(self, *args, **kwargs):
        self._reject_stage_args("create_dataset", args, kwargs)
        logger.info("Running prepare() (BaseSystem stub). Nothing done.")

    def collect_stats(self, *args, **kwargs):
        self._reject_stage_args("collect_stats", args, kwargs)
        logger.info(
            "Collecting stats | exp_dir=%s stats_dir=%s",
            getattr(self.train_config, "exp_dir", None),
            getattr(self.train_config, "stats_dir", None),
        )
        return collect_stats(self.train_config)

    def train(self, *args, **kwargs):
        self._reject_stage_args("train", args, kwargs)
        model_target = None
        if self.train_config is not None and hasattr(self.train_config, "model"):
            model_cfg = self.train_config.model
            if isinstance(model_cfg, DictConfig):
                model_target = model_cfg.get("_target_")
        logger.info(
            "Training start | exp_dir=%s model=%s",
            getattr(self.train_config, "exp_dir", None),
            model_target or "<unknown>",
        )
        return train(self.train_config)

    def evaluate(self, *args, **kwargs):
        self._reject_stage_args("evaluate", args, kwargs)
        # Backward-compat shim if someone calls evaluate directly.
        self.infer()
        return self.measure()

    def infer(self, *args, **kwargs):
        self._reject_stage_args("infer", args, kwargs)
        logger.info(
            "Inference start | decode_dir=%s",
            getattr(self.infer_config, "decode_dir", None),
        )
        return inference(self.infer_config)

    def measure(self, *args, **kwargs):
        self._reject_stage_args("measure", args, kwargs)
        logger.info(
            "Scoring start | metric_config=%s",
            self.metric_config is not None,
        )
        result = score(self.metric_config)
        logger.info("Scoring results: %s", result)
        return result

    def publish(self, *args, **kwargs):
        self._reject_stage_args("publish", args, kwargs)
        logger.info("Running publish() (BaseSystem stub). Nothing done.")
