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
