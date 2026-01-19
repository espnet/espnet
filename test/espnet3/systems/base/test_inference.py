from pathlib import Path

import pytest
from omegaconf import OmegaConf

import espnet3.systems.base.inference as inference_mod


def dummy_output_fn(*, data, model_output, idx):
    return {"idx": idx, "hyp": "h", "ref": "r"}
from espnet3.systems.base.inference_runner import InferenceRunner


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
            "output_keys": ["hyp", "ref"],
            "idx_key": "idx",
            "mock_dataset_length": 2,
        }
    )
    results = [
        {"idx": 0, "hyp": "h0", "ref": "r0"},
        {"idx": 1, "hyp": "h1", "ref": "r1"},
    ]
    calls = {}

    def fake_set_parallel(arg):
        calls["parallel"] = arg

    def runner_factory(provider, *, async_mode=False, **_kwargs):
        return DummyRunner(provider, async_mode=async_mode, results=results)

    monkeypatch.setattr(inference_mod, "set_parallel", fake_set_parallel)
    monkeypatch.setattr(inference_mod, "InferenceProvider", DummyProvider)
    monkeypatch.setattr(inference_mod, "InferenceRunner", runner_factory)

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
            "output_keys": ["hyp", "ref"],
            "idx_key": "idx",
            "mock_dataset_length": 1,
        }
    )

    def runner_factory(provider, *, async_mode=False, **_kwargs):
        return DummyRunner(provider, async_mode=async_mode, results=None)

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)
    monkeypatch.setattr(inference_mod, "InferenceProvider", DummyProvider)
    monkeypatch.setattr(inference_mod, "InferenceRunner", runner_factory)

    with pytest.raises(RuntimeError, match="Async inference is not supported"):
        inference_mod.inference(cfg)
