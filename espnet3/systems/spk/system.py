"""Speaker identification system for ESPnet3.

This module adds speaker-specific stages on top of the base system,
including dataset creation hooks. Unlike ASR, no tokenizer training
is required.
"""

import logging
import time
from importlib import import_module

from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


def load_function(path):
    """Load a callable from a dotted module path.

    Args:
        path: Dotted module path (e.g., ``package.module.function``).

    Returns:
        Callable referenced by the path.

    Raises:
        (Exception): Propagated import or attribute lookup errors.
    """
    module_path, func_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, func_name)


class SPKSystem(BaseSystem):
    """Speaker identification system.

    This system supports speaker verification, identification, and
    diarization workflows. It delegates training to the base system
    and can be configured entirely via YAML.

    Stage log paths follow the base system defaults:
        | Stage          | Path reference             |
        |---             |---                         |
        | create_dataset | training_config.data_dir   |
        | collect_stats  | training_config.stats_dir  |
        | train          | training_config.exp_dir    |
        | infer          | inference_config.inference_dir |
        | measure        | metrics_config.inference_dir |
    """

    def __init__(
        self,
        training_config=None,
        inference_config=None,
        metrics_config=None,
        **kwargs,
    ) -> None:
        """Initialize the speaker system."""
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            **kwargs,
        )

    def create_dataset(self, *args, **kwargs):
        """Create datasets using the configured helper function.

        The callable is resolved from ``training_config.create_dataset.func`` and
        invoked with the remaining configuration values.

        Raises:
            RuntimeError: If the configuration does not specify a function.
        """
        self._reject_stage_args("create_dataset", args, kwargs)
        logger.info("SPKSystem.create_dataset(): starting dataset creation process")
        start = time.perf_counter()
        config = getattr(self.training_config, "create_dataset", None)
        if config is None or not getattr(config, "func", None):
            raise RuntimeError(
                "training_config.create_dataset.func must be set to run create_dataset"
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
