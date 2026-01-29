from pathlib import Path

import pytest
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import OmegaConf

import espnet3.systems.base.inference as inference_mod
from espnet3.systems.base.inference_runner import InferenceRunner


def dummy_output_fn(*, data, model_output, idx):
    return {"idx": idx, "hyp": "h", "ref": "r"}


class DummyProvider:
    def __init__(self, config, **kwargs):
        self.config = config

    def build_dataset(self, _config):
        return [None] * self.config.mock_dataset_length


class DummyRunner(InferenceRunner):
    def __init__(self, provider, *, async_mode=False, results=None):
        super().__init__(provider, async_mode=async_mode)
        self._results = results if results is not None else None

    @staticmethod
    def forward(idx, *, dataset, model, **env):
        return {"idx": idx, "hyp": "h", "ref": "r"}

    def __call__(self, indices):
        return self._results


runner_factory_results = None


def runner_factory(provider, *, async_mode=False, **_kwargs):
    return DummyRunner(provider, async_mode=async_mode, results=runner_factory_results)


class CaptureProvider:
    last_params = None

    def __init__(self, config, *, params):
        CaptureProvider.last_params = params
        self.config = config

    def build_dataset(self, _config):
        return [None] * self.config.mock_dataset_length


class CaptureRunner:
    results = None

    def __init__(self, provider, **_kwargs):
        self.provider = provider

    def __call__(self, _indices):
        return CaptureRunner.results

    idx_key = "idx"


def _instantiate_without_config(*args, **kwargs):
    if not args:
        raise RuntimeError("instantiate requires a config argument")
    config = args[0]
    rest_args = args[1:]
    if "config" in kwargs:
        cfg = kwargs.pop("config")
        return DummyProvider(cfg, **kwargs)
    if "provider" in kwargs:
        return runner_factory(kwargs["provider"], async_mode=kwargs.get("async_mode"))
    return hydra_instantiate(config, *rest_args, **kwargs)


def _read_scp(path: Path):
    return path.read_text(encoding="utf-8").splitlines()


def test_inference_writes_scp_outputs(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "infer_dir": str(tmp_path / "infer"),
            "dataset": {"test": [{"name": "test_a"}, {"name": "test_b"}]},
            "input_key": "speech",
            "output_fn": f"{__name__}.dummy_output_fn",
            "idx_key": "idx",
            "mock_dataset_length": 2,
            "provider": {"_target_": f"{__name__}.DummyProvider"},
            "runner": {"_target_": f"{__name__}.DummyRunner"},
        }
    )
    results = [
        {"idx": 0, "hyp": "h0", "ref": "r0"},
        {"idx": 1, "hyp": "h1", "ref": "r1"},
    ]
    calls = {}

    def fake_set_parallel(arg):
        calls["parallel"] = arg

    global runner_factory_results
    runner_factory_results = results

    monkeypatch.setattr(inference_mod, "set_parallel", fake_set_parallel)
    monkeypatch.setattr(inference_mod, "instantiate", _instantiate_without_config)

    inference_mod.inference(cfg)

    assert calls["parallel"] == {"env": "local"}
    for test_name in ("test_a", "test_b"):
        base = tmp_path / "infer" / test_name
        assert _read_scp(base / "hyp.scp") == ["0 h0", "1 h1"]
        assert _read_scp(base / "ref.scp") == ["0 r0", "1 r1"]


def test_inference_rejects_async_results(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "infer_dir": str(tmp_path / "infer"),
            "dataset": {"test": [{"name": "test_a"}]},
            "input_key": "speech",
            "output_fn": f"{__name__}.dummy_output_fn",
            "idx_key": "idx",
            "mock_dataset_length": 1,
            "provider": {"_target_": f"{__name__}.DummyProvider"},
            "runner": {"_target_": f"{__name__}.DummyRunner"},
        }
    )

    global runner_factory_results
    runner_factory_results = None

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)
    monkeypatch.setattr(inference_mod, "instantiate", _instantiate_without_config)

    with pytest.raises(RuntimeError, match="Async inference is not supported"):
        inference_mod.inference(cfg)


def test_inference_passes_provider_params(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "infer_dir": str(tmp_path / "infer"),
            "dataset": {"test": [{"name": "test_a"}]},
            "input_key": "speech",
            "output_fn": f"{__name__}.dummy_output_fn",
            "idx_key": "idx",
            "mock_dataset_length": 1,
            "provider": {
                "_target_": f"{__name__}.CaptureProvider",
                "params": {"beam": 5, "lang": "en"},
            },
            "runner": {"_target_": f"{__name__}.CaptureRunner"},
        }
    )
    results = [{"idx": 0, "hyp": "h0", "ref": "r0"}]
    CaptureProvider.last_params = None
    CaptureRunner.results = results

    def fake_instantiate(obj, **kwargs):
        if "config" in kwargs:
            return CaptureProvider(kwargs["config"], params=kwargs["params"])
        return CaptureRunner(kwargs["provider"])

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)
    monkeypatch.setattr(inference_mod, "instantiate", fake_instantiate)

    inference_mod.inference(cfg)

    assert CaptureProvider.last_params == {
        "beam": 5,
        "lang": "en",
        "input_key": "speech",
        "output_fn_path": f"{__name__}.dummy_output_fn",
    }
