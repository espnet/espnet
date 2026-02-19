"""Inference-time provider helpers for dataset/model construction."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.utils.logging_utils import log_instance_dict

logger = logging.getLogger(__name__)
_LOGGED_ENV = False


class InferenceProvider(EnvironmentProvider, ABC):
    """Provider base class tailored for inference-time datasets/models.

    Subclasses implement dataset/model creation for local execution or
    per-worker setup. Instances cache a local environment for reuse.
    """

    def __init__(self, config: DictConfig, params: Optional[Dict[str, Any]] = None):
        """Initialize the provider and prebuild the local environment.

        Args:
            config: Configuration for dataset/model construction.
            params: Optional extra entries merged into the environment.
        """
        super().__init__(config)
        self.params = params or {}
        # Build once for local execution to avoid redundant IO
        self._local_env = self.build_worker_setup_fn()()

    @staticmethod
    @abstractmethod
    def build_dataset(config: DictConfig):
        """Create a dataset object from config.

        Args:
            config: Configuration for dataset construction.

        Returns:
            Dataset-like object used for inference.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_model(config: DictConfig):
        """Create a model object from config.

        Args:
            config: Configuration for model construction.

        Returns:
            Model-like object used for inference.
        """
        raise NotImplementedError

    def build_env_local(self) -> Dict[str, Any]:
        """Return the already constructed local environment.

        Returns:
            Mapping with dataset/model and any extra params.
        """
        env = dict(self._local_env)
        self._log_env(env)
        return env

    def build_worker_setup_fn(self):
        """Return a setup function that rebuilds the env per worker.

        Returns:
            Callable that constructs a fresh environment when invoked.
        """
        config = self.config
        params = dict(self.params)

        def setup() -> Dict[str, Any]:
            env = {
                "dataset": self.build_dataset(config),
                "model": self.build_model(config),
            }
            env.update(params)
            self._log_env(env)
            return env

        return setup

    def _log_env(self, env: Dict[str, Any]) -> None:
        global _LOGGED_ENV
        if _LOGGED_ENV:
            return
        _LOGGED_ENV = True
        log_instance_dict(logger, kind="Env", entries=env, max_depth=2)
