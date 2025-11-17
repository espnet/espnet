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
        self, train_config: DictConfig, eval_config: DictConfig = None
    ) -> None:
        self.train_config = train_config
        self.eval_config = eval_config
        self.exp_dir = Path(train_config.exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Stage stubs (override in subclasses if needed)
    # ---------------------------------------------------------
    def create_dataset(self):
        logger.info("Running prepare() (BaseSystem stub). Nothing done.")

    def collect_stats(self):
        return collect_stats(self.train_config)

    def train(self):
        return train(self.train_config)

    def evaluate(self):
        self.decode()
        return self.score()

    def decode(self):
        return inference(self.eval_config)

    def score(self):
        result = score(self.eval_config)
        logger.info("Scoring results: %s", result)
        return result

    def publish(self):
        logger.info("Running publish() (BaseSystem stub). Nothing done.")
