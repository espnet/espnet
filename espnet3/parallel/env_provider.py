"""Base interface for environment providers in ESPnet3 runners."""

from typing import Any, Callable, Dict

from omegaconf import DictConfig


class EnvironmentProvider:
    """A base interface to build and inject per-process environments.

    This class separates responsibilities for constructing shared
    resources (e.g., dataset/model/tokenizer) that need to be created
    either once on the driver or once per worker process.

    Subclasses should implement ``build_env_local`` and
    ``make_worker_setup_fn`` to define how the environment is built
    in local (driver) execution and distributed (worker) execution.

    Args:
        config (DictConfig): A Hydra/OmegaConf configuration object
            that contains all parameters needed to build the environment.

    Notes:
        - The environment returned by these builders must be a plain
          dictionary of lightweight, pickleable objects or handles
          that are safe to share within a worker process.
        - For distributed runs, heavy initialization should be done inside
          the worker setup function so each worker constructs its own copy.
    """

    # TODO(Masao) Add detailed description on Runner/Provider in the document.

    def __init__(self, config: DictConfig):
        """Initialize EnvironmentProvider object."""
        self.config = config

    def build_env_local(self) -> Dict[str, Any]:
        """Build the environment once on the driver for local execution.

        This method is called in purely local runs (no Dask). Typical usage
        is to instantiate dataset/model/tokenizer directly and return them
        as a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing environment objects
            (e.g., ``{"dataset": ds, "model": md, ...}``).

        Raises:
            NotImplementedError: If the subclass does not override this method.

        Example:
            >>> class MyProvider(EnvironmentProvider):
            ...     def build_env_local(self):
            ...         ds = build_dataset(self.config)
            ...         md = build_model(self.config)
            ...         return {"dataset": ds, "model": md}
        """
        raise NotImplementedError

    def make_worker_setup_fn(self) -> Callable[[], Dict[str, Any]]:
        """Create a worker setup function for distributed execution.

        The returned callable will be executed **once per worker** to build
        that worker's environment and cached via a Dask ``WorkerPlugin``.
        The resulting dictionary is later injected into user functions by
        name-matching of keyword parameters.

        Returns:
            Callable[[], Dict[str, Any]]: A zero-arg function that returns an
            environment dictionary when called on a worker.

        Raises:
            NotImplementedError: If the subclass does not override this method.

        Example:
            >>> class MyProvider(EnvironmentProvider):
            ...     def make_worker_setup_fn(self):
            ...         cfg = self.config
            ...         def setup():
            ...             ds = build_dataset(cfg)
            ...             md = build_model(cfg)
            ...             return {"dataset": ds, "model": md}
            ...         return setup  # return setup function!
        """
        raise NotImplementedError
