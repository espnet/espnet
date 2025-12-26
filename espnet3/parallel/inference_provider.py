from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from espnet3.parallel.env_provider import EnvironmentProvider


class InferenceProvider(EnvironmentProvider, ABC):
    """Provider base class tailored for inference-time datasets/models."""

    def __init__(self, config: DictConfig, *, params: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.params = params or {}
        # Build once for local execution to avoid redundant IO
        self._local_env = {
            "dataset": self.build_dataset(config),
            "model": self.build_model(config),
            **self.params,
        }

    @staticmethod
    @abstractmethod
    def build_dataset(cfg: DictConfig):
        """Create a dataset object from config."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_model(cfg: DictConfig):
        """Create a model object from config."""
        raise NotImplementedError

    def build_env_local(self) -> Dict[str, Any]:
        """Return the already constructed local environment."""
        return dict(self._local_env)

    def make_worker_setup_fn(self):
        """Return a setup function that rebuilds the env per worker."""
        cfg = self.config
        params = dict(self.params)

        def setup() -> Dict[str, Any]:
            env = {
                "dataset": self.build_dataset(cfg),
                "model": self.build_model(cfg),
            }
            env.update(params)
            return env

        return setup
