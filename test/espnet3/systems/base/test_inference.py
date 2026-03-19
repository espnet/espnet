from pathlib import Path

import pytest
from omegaconf import OmegaConf

import espnet3.systems.base.inference as inference_mod
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import InferenceRunner


def dummy_output_fn(*, data, model_output, idx):
    return {"idx": idx, "hyp": "h", "ref": "r"}


class DummyProvider(InferenceProvider):
    def __init__(self, inference_config, params):
        super().__init__(inference_config)
        self.params = params

    def build_dataset(self, _config):
        return [None] * _config.mock_dataset_length


class DummyRunner(InferenceRunner):
    results = None

    def __init__(self, provider, *, async_mode=False, results=None, **kwargs):
        super().__init__(provider, async_mode=async_mode, **kwargs)
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

    def __init__(self, inference_config, *, params):
        super().__init__(inference_config)
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

    def resolve_idx_key(self, _output):
        return self.idx_key


def _read_scp(path: Path):
    return path.read_text(encoding="utf-8").splitlines()


def test_inference_writes_scp_outputs(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "inference_dir": str(tmp_path / "infer"),
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

    inference_mod.infer(cfg)

    assert calls["parallel"] == {"env": "local"}
    for test_name in ("test_a", "test_b"):
        base = tmp_path / "infer" / test_name
        assert _read_scp(base / "hyp.scp") == ["0 h0", "1 h1"]
        assert _read_scp(base / "ref.scp") == ["0 r0", "1 r1"]


def test_inference_rejects_async_results(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "inference_dir": str(tmp_path / "infer"),
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
        inference_mod.infer(cfg)


def test_inference_passes_provider_params(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "inference_dir": str(tmp_path / "infer"),
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

    inference_mod.infer(cfg)

    assert CaptureProvider.last_params == {
        "beam": 5,
        "lang": "en",
        "input_key": "speech",
        "output_fn_path": f"{__name__}.dummy_output_fn",
    }


def test_inference_without_output_fn_uses_model_output(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "inference_dir": str(tmp_path / "infer"),
            "dataset": {"test": [{"name": "test_a"}]},
            "input_key": "speech",
            "mock_dataset_length": 2,
            "provider": {"_target_": f"{__name__}.DummyProvider"},
            "runner": {"_target_": f"{__name__}.DummyRunner"},
        }
    )
    DummyRunner.results = [
        {"utt_id": "utt1", "hyp": "h1"},
        {"utt_id": "utt2", "hyp": "h2"},
    ]

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)

    inference_mod.infer(cfg)

    assert _read_scp(tmp_path / "infer" / "test_a" / "hyp.scp") == [
        "utt1 h1",
        "utt2 h2",
    ]


def test_inference_without_idx_key_uses_default_utt_id(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "inference_dir": str(tmp_path / "infer"),
            "dataset": {"test": [{"name": "test_a"}]},
            "input_key": "speech",
            "output_fn": f"{__name__}.dummy_output_fn",
            "mock_dataset_length": 1,
            "provider": {"_target_": f"{__name__}.DummyProvider"},
            "runner": {"_target_": f"{__name__}.DummyRunner"},
        }
    )
    DummyRunner.results = [{"utt_id": "utt1", "hyp": "h1"}]

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)

    inference_mod.infer(cfg)

    assert _read_scp(tmp_path / "infer" / "test_a" / "hyp.scp") == ["utt1 h1"]


def test_inference_with_explicit_idx_key_override(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "inference_dir": str(tmp_path / "infer"),
            "dataset": {"test": [{"name": "test_a"}]},
            "input_key": "speech",
            "idx_key": "sample_id",
            "mock_dataset_length": 1,
            "provider": {"_target_": f"{__name__}.DummyProvider"},
            "runner": {"_target_": f"{__name__}.DummyRunner"},
        }
    )
    DummyRunner.results = [{"sample_id": "sample-1", "hyp": "h1"}]

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)

    inference_mod.infer(cfg)

    assert _read_scp(tmp_path / "infer" / "test_a" / "hyp.scp") == ["sample-1 h1"]


def test_inference_with_explicit_output_keys_filters_scp_outputs(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "inference_dir": str(tmp_path / "infer"),
            "dataset": {"test": [{"name": "test_a"}]},
            "input_key": "speech",
            "output_keys": ["hyp"],
            "mock_dataset_length": 1,
            "provider": {"_target_": f"{__name__}.DummyProvider"},
            "runner": {"_target_": f"{__name__}.DummyRunner"},
        }
    )
    DummyRunner.results = [{"utt_id": "utt1", "hyp": "h1", "ref": "r1"}]

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)

    inference_mod.infer(cfg)

    base = tmp_path / "infer" / "test_a"
    assert (base / "hyp.scp").exists()
    assert not (base / "ref.scp").exists()


def test_inference_without_output_fn_requires_utterance_id(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "inference_dir": str(tmp_path / "infer"),
            "dataset": {"test": [{"name": "test_a"}]},
            "input_key": "speech",
            "mock_dataset_length": 1,
            "provider": {"_target_": f"{__name__}.DummyProvider"},
            "runner": {"_target_": f"{__name__}.DummyRunner"},
        }
    )
    DummyRunner.results = [{"hyp": "h1"}]

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)

    with pytest.raises(ValueError, match="sample identifier key"):
        inference_mod.infer(cfg)
