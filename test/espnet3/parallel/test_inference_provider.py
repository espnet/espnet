from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet3.parallel.inference_provider import InferenceProvider


class CountingProvider(InferenceProvider):
    dataset_calls = 0
    model_calls = 0

    @staticmethod
    def build_dataset(cfg):
        CountingProvider.dataset_calls += 1
        return {"name": cfg.name, "call": CountingProvider.dataset_calls}

    @staticmethod
    def build_model(cfg):
        CountingProvider.model_calls += 1
        return {"name": cfg.name, "call": CountingProvider.model_calls}


def test_build_env_local_uses_cached_env_and_params():
    CountingProvider.dataset_calls = 0
    CountingProvider.model_calls = 0
    cfg = OmegaConf.create({"name": "test"})
    provider = CountingProvider(cfg, params={"flag": True})

    env = provider.build_env_local()

    assert env["dataset"]["name"] == "test"
    assert env["model"]["name"] == "test"
    assert env["flag"] is True
    assert CountingProvider.dataset_calls == 1
    assert CountingProvider.model_calls == 1

    env["dataset"] = "mutated"
    assert provider.build_env_local()["dataset"] != "mutated"


def test_build_worker_setup_fn_rebuilds_env_and_captures_params():
    CountingProvider.dataset_calls = 0
    CountingProvider.model_calls = 0
    cfg = OmegaConf.create({"name": "worker"})
    provider = CountingProvider(cfg, params={"token": "v1"})

    setup = provider.build_worker_setup_fn()
    provider.params["token"] = "v2"

    env1 = setup()
    env2 = setup()

    assert env1["token"] == "v1"
    assert env2["token"] == "v1"
    assert env1["dataset"] != env2["dataset"]
    assert env1["model"] != env2["model"]
    assert CountingProvider.dataset_calls == 3
    assert CountingProvider.model_calls == 3


class _FakeDataset:
    """Stand-in dataset built through a real Hydra ``_target_``."""

    def __init__(self, split: str):
        self.split = split


class _FakeModel:
    """Stand-in model built through a real Hydra ``_target_``."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale


class HydraTargetProvider(InferenceProvider):
    """Provider whose build_dataset/build_model drive real hydra.utils.instantiate.

    Mirrors how a real recipe provider resolves ``dataset``/``model`` blocks
    (``_target_`` + instantiate), and normalizes a plain ``dict`` config to a
    ``DictConfig`` first -- the dict-vs-DictConfig branch this module leaves to
    subclasses (unlike espnet3.systems.base.inference_provider.InferenceProvider,
    which normalizes at the base-class level).
    """

    @staticmethod
    def build_dataset(config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        return instantiate(config.dataset)

    @staticmethod
    def build_model(config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        return instantiate(config.model)


def test_build_env_local_instantiates_real_hydra_targets_from_dictconfig():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "_target_": f"{__name__}._FakeDataset",
                "split": "test",
            },
            "model": {
                "_target_": f"{__name__}._FakeModel",
                "scale": 2.0,
            },
        }
    )
    provider = HydraTargetProvider(cfg, params={})

    env = provider.build_env_local()

    assert isinstance(env["dataset"], _FakeDataset)
    assert env["dataset"].split == "test"
    assert isinstance(env["model"], _FakeModel)
    assert env["model"].scale == 2.0


def test_build_env_local_instantiates_real_hydra_targets_from_plain_dict():
    cfg = {
        "dataset": {
            "_target_": f"{__name__}._FakeDataset",
            "split": "valid",
        },
        "model": {
            "_target_": f"{__name__}._FakeModel",
            "scale": 3.0,
        },
    }
    provider = HydraTargetProvider(cfg, params={})

    env = provider.build_env_local()

    assert isinstance(env["dataset"], _FakeDataset)
    assert env["dataset"].split == "valid"
    assert isinstance(env["model"], _FakeModel)
    assert env["model"].scale == 3.0
