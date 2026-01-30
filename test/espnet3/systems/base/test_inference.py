from pathlib import Path

import pytest
from omegaconf import OmegaConf

import espnet3.systems.base.inference as inference_mod
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import InferenceRunner


def dummy_output_fn(*, data, model_output, idx):
    return {"idx": idx, "hyp": "h", "ref": "r"}


class DummyProvider(InferenceProvider):
    def __init__(self, infer_config, params):
        super().__init__(infer_config)
        self.params = params

    def build_dataset(self, _config):
        return [None] * _config.mock_dataset_length


class DummyRunner(InferenceRunner):
    results = None

    def __init__(self, provider, *, async_mode=False, results=None, **_kwargs):
        super().__init__(provider, async_mode=async_mode)
        if results is not None:
            self._results = results
        else:
            self._results = DummyRunner.results

    @staticmethod
    def forward(idx, *, dataset, model, **env):
        return {"idx": idx, "hyp": "h", "ref": "r"}

    def __call__(self, indices):
        return self._results


class CaptureProvider(InferenceRunner):
    last_params = None

    def __init__(self, infer_config, *, params):
        super().__init__(infer_config)
        CaptureProvider.last_params = params

    def build_dataset(self, _config):
        return [None] * _config.mock_dataset_length


class CaptureRunner:
    results = None

    def __init__(self, provider, **_kwargs):
        self.provider = provider

    def __call__(self, _indices):
        return CaptureRunner.results

    idx_key = "idx"


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

    DummyRunner.results = results

    monkeypatch.setattr(inference_mod, "set_parallel", fake_set_parallel)

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

    DummyRunner.results = None

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)

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

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)

    inference_mod.inference(cfg)

    assert CaptureProvider.last_params == {
        "beam": 5,
        "lang": "en",
        "input_key": "speech",
        "output_fn_path": f"{__name__}.dummy_output_fn",
    }
