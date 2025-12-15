"""InferenceProvider for dataset/model inference setup."""

from typing import Any, Callable, Dict

from omegaconf import DictConfig

from espnet3.runner.env_provider import EnvironmentProvider


class InferenceProvider(EnvironmentProvider):
    """EnvironmentProvider specialized for dataset/model inference setup.

    This implementation focuses on constructing just the ``dataset`` and
    ``model`` and returning them as environment entries. It is suitable
    for both local and distributed execution.

    Design:
        - ``build_*`` helpers are defined as ``@staticmethod``/class methods
          so they are easily serializable and reusable on workers.
        - The worker setup function must **not** capture ``self`` to remain
          pickle-safe for Dask.

    Args:
        config (DictConfig): Hydra configuration used to build dataset/model.
        params (Dict[str, Any] | None): Optional additional key-value pairs
            that will be merged into the returned environment (e.g., device,
            tokenizer, beam size).

    Notes:
        - Subclasses must implement ``build_dataset`` and ``build_model``.
        - ``self.config.update(self.params)`` allows lightweight overrides
          (e.g., runtime overrides) but avoid mutating deep structures unless
          intended.
    """

    # TODO(Masao) Add detailed description on Runner/Provider in the document.

    def __init__(self, config: DictConfig, *, params: Dict[str, Any] | None = None):
        """Initialize InferenceProvider object."""
        super().__init__(config)
        self.params = params or {}
        self.config.update(self.params)

    def build_env_local(self) -> Dict[str, Any]:
        """Build the environment once on the driver for local inference.

        Returns:
            Dict[str, Any]: Environment dict with at least:
                - ``"dataset"``: The instantiated dataset.
                - ``"model"``: The instantiated model.
              Any additional fields from ``params`` are also included.

        Example:
            >>> provider = InferenceProvider(cfg, params={"device": "cuda"})
            >>> env = provider.build_env_local()
            >>> env.keys()
            dict_keys(["dataset", "model", "device"])
        """
        config = self.config

        dataset = self.build_dataset(config)
        model = self.build_model(config)

        env = {"dataset": dataset, "model": model}
        env.update(self.params)
        return env

    def make_worker_setup_fn(self) -> Callable[[], Dict[str, Any]]:
        """Return a Dask worker setup function that builds dataset/model.

        The returned function is executed once per worker process and must
        not capture ``self``. It closes over the immutable config and params
        snapshot, then constructs the environment on each worker.

        Returns:
            Callable[[], Dict[str, Any]]: A zero-argument setup function that
            returns ``{"dataset": ..., "model": ..., **params}``.

        Example:
            >>> provider = InferenceProvider(cfg, params={"device": "cuda:0"})
            >>> setup_fn = provider.make_worker_setup_fn()
            >>> env = setup_fn()
            >>> "dataset" in env and "model" in env
            True
        """
        config = self.config
        params = dict(self.params)
        cls = self.__class__

        def setup_fn() -> Dict[str, Any]:
            dataset = cls.build_dataset(config)
            model = cls.build_model(config)
            env = {"dataset": dataset, "model": model}
            env.update(params)
            return env

        return setup_fn

    # Implement the following functions in the subclass
    @staticmethod
    def build_dataset(config: DictConfig):
        """Construct and return the dataset instance.

        Implemented by subclasses to build a dataset from ``config``.
        During parallel or distributed execution, the ``config`` object passed here
        is the configuration that the user passed when instantiating the class.

        Args:
            config (DictConfig): Configuration object for dataset
                parameters (e.g., data directory, preprocessing pipeline,
                features, split).

        Returns:
            Any: Dataset object (type defined by subclass).

        Raises:
            NotImplementedError: Always in the base class; implement in subclass.

        Example:
            >>> # Minimal sketch; actual keys depend on your subclass
            >>> from omegaconf import OmegaConf
            >>> cfg = OmegaConf.create({
            >>>     "dataset": {"path": "data/test", "split": "test"}
            >>> })
            >>> ds = MyInferenceProvider.build_dataset(cfg)

        Notes:
            - Keep dataset initialization lightweight by using lazy loading or
              memory mapping when possible.
            - Rely on fields already present in ``config`` instead of reading
              global state whenever possible.
        """
        raise NotImplementedError(
            "Implement build_dataset(config=self.config) in subclass."
        )

    @staticmethod
    def build_model(config: DictConfig):
        """Construct and return the model instance.

        Implemented by subclasses to build a model from ``config``.
        During parallel or distributed execution, the ``config`` object passed here
        is the configuration that the user passed when instantiating the class.

        Args:
            cfg (DictConfig): Configuration.

        Returns:
            Any: Model object (type defined by subclass).

        Raises:
            NotImplementedError: Always in the base class; implement in subclass.

        Example:
            >>> # Minimal sketch; actual keys depend on your subclass
            >>> from omegaconf import OmegaConf
            >>> cfg = OmegaConf.create({
            >>>     "model": {"checkpoint": "exp/model.pth", "device": "cpu"}
            >>> })
            >>> model = MyInferenceProvider.build_model(cfg)  # doctest: +SKIP

        Notes:
            - This method should handle **loading weights** and placing the
              model on the appropriate device.
            - Do not perform training/optimization here, this is for inference
              setup only.
        """
        raise NotImplementedError("Implement build_model(self.config) in subclass.")
