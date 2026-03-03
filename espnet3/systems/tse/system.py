"""TSE system implementation and training helpers.

This module adds TSE-specific stages on top of the base system, including
dataset creation hooks.
"""

import logging
import time

from espnet3.systems.asr.system import load_function
from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


class TSESystem(BaseSystem):
    """TSE-specific system.

    This system adds:
      - Dataset creation using the configured helper function.
    """

    def create_dataset(self, *args, **kwargs):
        """Create datasets using the configured helper function.

        The callable is resolved from ``train_config.create_dataset.func`` and
        invoked with the remaining configuration values.

        Raises:
            RuntimeError: If the configuration does not specify a function.
        """
        self._reject_stage_args("create_dataset", args, kwargs)
        logger.info("TSESystem.create_dataset(): starting dataset creation process")
        start = time.perf_counter()
        config = getattr(self.train_config, "create_dataset", None)
        if config is None or not getattr(config, "func", None):
            raise RuntimeError(
                "train_config.create_dataset.func must be set to run create_dataset"
            )
        fn = load_function(config.func)
        extra = {k: v for k, v in config.items() if k != "func"}
        logger.info("Creating dataset with function %s", config.func)
        result = fn(**extra)
        logger.info(
            "Dataset creation completed in %.2fs using %s",
            time.perf_counter() - start,
            config.func,
        )
        return result

    def train(self, *args, **kwargs):
        """Train the TSE model.

        Raises:
            RuntimeError: If ``train_config.dataset_dir`` is not set.
        """
        self._reject_stage_args("train", args, kwargs)
        logger.info("TSESystem.train(): starting training process")

        dataset_dir = getattr(self.train_config, "dataset_dir", None)
        if dataset_dir is None:
            raise RuntimeError("train_config.dataset_dir must be set for training.")

        return super().train()
