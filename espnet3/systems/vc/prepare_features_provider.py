"""Provider for the parallel ``prepare_features`` stage of ``VCSystem``."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.components.data.dataset_module import instantiate_dataset_reference
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.systems.vc.prepare_features_runner import PoolFeatureCache


class PrepareFeaturesProvider(EnvironmentProvider):
    """Build the per-worker environment for ``PrepareFeaturesRunner``.

    Each worker needs the audio dataset (to read waveforms), the frozen
    feature encoder (a Hydra ``_target_`` such as
    :class:`espnet3.systems.vc.models.knnvc.wavlm_encoder.WavLMEncoder`), the
    pool-key-to-indices map used to build prematching pools, and the stage
    options. Heavy objects (the encoder) are constructed inside
    ``build_worker_setup_fn`` so every Dask worker owns its own copy.

    Dataset contract (checked at construction): the object returned by the
    recipe's ``Dataset(**data_src_args)`` must, in addition to ``__len__`` and
    ``__getitem__`` returning ``{"speech": <1-D float32 waveform>}``, expose

    - ``get_pool_key(idx) -> str``: prematching-pool key of item ``idx``,
      obtained without loading audio. Utterances sharing a key form each
      other's matching pool (a speaker id, or a speaker/chapter id to mimic
      the official kNN-VC script, which pools per corpus directory);
    - ``get_feature_name(idx) -> str``: relative path (without suffix) under
      the feature output directory where item ``idx``'s features are written,
      e.g. ``train-clean-100/103/1240/103-1240-0000``.

    Args:
        config: Training config; only used to resolve ``recipe_dir`` for
            local dataset modules.
        params: Stage parameters:

            - ``dataset`` (dict): one dataset reference entry
              (``data_src`` / ``data_src_args``);
            - ``encoder`` (dict): Hydra config of the encoder module;
            - ``features_dir`` (str): where feature files are written;
            - ``prematch`` (bool): apply kNN prematching within each pool;
            - ``topk`` (int): ``k`` used for prematching;
            - ``device`` (str | None): device for the encoder, ``None`` for
              ``cuda`` when available else ``cpu``.
    """

    REQUIRED_DATASET_METHODS = ("get_pool_key", "get_feature_name")

    def __init__(self, config: DictConfig, params: Dict[str, Any] | None = None):
        """Store the stage parameters and validate the dataset contract."""
        super().__init__(config)
        self.params = dict(params or {})
        for key in ("dataset", "encoder", "features_dir"):
            if self.params.get(key) is None:
                raise RuntimeError(
                    f"prepare_features requires `{key}`; check "
                    "training_config.prepare_features."
                )
        self.recipe_dir = getattr(config, "recipe_dir", None) if config else None
        # Fail fast on the driver if the recipe dataset lacks the stage methods.
        # The instance is reused by the stage (to plan shards by pool key) and
        # by `build_env_local`, so a corpus of this size is indexed once on the
        # driver. Dask workers build their own in `build_worker_setup_fn`.
        self.dataset = self.build_dataset(self.params, self.recipe_dir)
        self.validate_dataset(self.dataset)

    @classmethod
    def validate_dataset(cls, dataset) -> None:
        """Raise ``TypeError`` if ``dataset`` misses a required stage method."""
        missing = [
            name
            for name in cls.REQUIRED_DATASET_METHODS
            if not callable(getattr(dataset, name, None))
        ]
        if missing:
            raise TypeError(
                f"{type(dataset).__name__} cannot be used by prepare_features: "
                f"missing method(s) {missing}. See PrepareFeaturesProvider."
            )

    @staticmethod
    def build_dataset(params: Dict[str, Any], recipe_dir):
        """Instantiate the audio dataset from ``params['dataset']``.

        Args:
            params: Stage parameters; ``params["dataset"]`` is one dataset
                reference entry (``data_src`` / ``data_src_args``).
            recipe_dir: Recipe root used to resolve a recipe-local dataset
                module when the entry has no ``data_src``.

        Returns:
            The instantiated dataset object.
        """
        dataset_config = params["dataset"]
        if OmegaConf.is_config(dataset_config):
            dataset_config = OmegaConf.to_container(dataset_config, resolve=True)
        return instantiate_dataset_reference(dataset_config, recipe_dir=recipe_dir)

    @staticmethod
    def build_encoder(params: Dict[str, Any], device: str):
        """Instantiate the encoder module on ``device``.

        Args:
            params: Stage parameters; ``params["encoder"]`` is a Hydra config
                with a ``_target_`` accepting a ``device`` keyword.
            device: Torch device string the encoder is built on.

        Returns:
            The instantiated encoder, expected to expose
            ``encode(speech, pad_to_hop)`` and a ``device`` attribute.
        """
        encoder_config = params["encoder"]
        if OmegaConf.is_config(encoder_config):
            encoder_config = OmegaConf.to_container(encoder_config, resolve=True)
        return instantiate(encoder_config, device=device)

    @staticmethod
    def build_pool_indices(dataset) -> Dict[str, List[int]]:
        """Group dataset indices by ``dataset.get_pool_key(idx)``.

        Args:
            dataset: Dataset implementing the ``prepare_features`` contract.

        Returns:
            Mapping of pool key to the dataset indices belonging to it, in
            index order. Utterances sharing a key prematch against each other.
        """
        pool_indices: Dict[str, List[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            pool_indices[str(dataset.get_pool_key(idx))].append(idx)
        return dict(pool_indices)

    @staticmethod
    def resolve_device(params: Dict[str, Any]) -> str:
        """Return the encoder device: explicit ``device`` or cuda-if-available.

        Args:
            params: Stage parameters; an explicit ``params["device"]`` wins.

        Returns:
            A torch device string such as ``"cuda"`` or ``"cpu"``.
        """
        device = params.get("device")
        if device:
            return str(device)
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _build_env(params: Dict[str, Any], recipe_dir, dataset=None) -> Dict[str, Any]:
        if dataset is None:
            dataset = PrepareFeaturesProvider.build_dataset(params, recipe_dir)
        PrepareFeaturesProvider.validate_dataset(dataset)
        device = PrepareFeaturesProvider.resolve_device(params)
        encoder = PrepareFeaturesProvider.build_encoder(params, device)
        return {
            "dataset": dataset,
            "model": encoder,
            "pool_indices": PrepareFeaturesProvider.build_pool_indices(dataset),
            "features_dir": str(params["features_dir"]),
            "prematch": bool(params.get("prematch", True)),
            "topk": int(params.get("topk", 4)),
            # Per-thread cache of one pool's features, shared by all `forward`
            # calls on that worker (see PrepareFeaturesRunner).
            "pool_cache": PoolFeatureCache(),
        }

    def build_env_local(self) -> Dict[str, Any]:
        """Build the environment once on the driver for local execution.

        Reuses the dataset already indexed in ``__init__`` instead of walking
        the corpus a second time.
        """
        return self._build_env(self.params, self.recipe_dir, dataset=self.dataset)

    def build_worker_setup_fn(self) -> Callable[[], Dict[str, Any]]:
        """Return a pickle-safe zero-arg setup function for Dask workers."""
        params = dict(self.params)
        recipe_dir = self.recipe_dir

        def setup() -> Dict[str, Any]:
            return PrepareFeaturesProvider._build_env(params, recipe_dir)

        return setup
