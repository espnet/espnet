from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

import espnet3.systems.base.inference as inference_mod
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import InferenceRunner


def dummy_output_fn(*, data, model_output, idx):
    return {"idx": idx, "hyp": "h", "ref": "r"}


def custom_npy_writer(*, value, output_path: Path, scale: float = 1.0) -> Path:
    target = output_path.with_suffix(".custom.npy")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, np.asarray(value) * scale)
    return target


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


def test_inference_rejects_test_entry_without_name(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "inference_dir": str(tmp_path / "infer"),
            "dataset": {"test": [{"data_src": "mini_an4/asr"}]},
            "input_key": "speech",
            "mock_dataset_length": 1,
            "provider": {"_target_": f"{__name__}.DummyProvider"},
            "runner": {"_target_": f"{__name__}.DummyRunner"},
        }
    )
    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)

    with pytest.raises(RuntimeError, match="must define non-empty `name`"):
        inference_mod.infer(cfg)


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


def test_inference_writes_artifacts_from_config(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "inference_dir": str(tmp_path / "infer"),
            "dataset": {"test": [{"name": "test_a"}]},
            "input_key": "speech",
            "mock_dataset_length": 1,
            "provider": {"_target_": f"{__name__}.DummyProvider"},
            "runner": {"_target_": f"{__name__}.DummyRunner"},
            "output_artifacts": {
                "audio": {"type": "wav", "sample_rate": 16000},
                "posterior": {"type": "npy"},
            },
        }
    )
    DummyRunner.results = [
        {
            "utt_id": "utt1",
            "audio": np.array([0.0, 0.1, -0.1], dtype=np.float32),
            "posterior": torch.tensor([1.0, 2.0]),
        }
    ]

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)

    inference_mod.infer(cfg)

    base = tmp_path / "infer" / "test_a"
    audio_path = Path(_read_scp(base / "audio.scp")[0].split(maxsplit=1)[1])
    posterior_path = Path(_read_scp(base / "posterior.scp")[0].split(maxsplit=1)[1])
    _, rate = sf.read(audio_path, dtype="float32")
    assert rate == 16000
    assert audio_path.suffix == ".wav"
    assert posterior_path.suffix == ".npy"
    assert np.array_equal(np.load(posterior_path), np.array([1.0, 2.0]))


def test_inference_writes_dict_artifacts_as_json(tmp_path, monkeypatch):
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
    DummyRunner.results = [
        {
            "utt_id": "utt1",
            "meta": {"frames": [1, 2, 3], "tag": "ok"},
        }
    ]

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)

    inference_mod.infer(cfg)

    base = tmp_path / "infer" / "test_a"
    path = Path(_read_scp(base / "meta.scp")[0].split(maxsplit=1)[1])
    assert path.suffix == ".json"
    assert path.read_text(encoding="utf-8").strip().startswith("{")


def test_inference_writes_artifacts_with_custom_writer(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "parallel": {"env": "local"},
            "inference_dir": str(tmp_path / "infer"),
            "dataset": {"test": [{"name": "test_a"}]},
            "input_key": "speech",
            "mock_dataset_length": 1,
            "provider": {"_target_": f"{__name__}.DummyProvider"},
            "runner": {"_target_": f"{__name__}.DummyRunner"},
            "output_artifacts": {
                "posterior": {
                    "writer": {
                        "_target_": f"{__name__}.custom_npy_writer",
                        "scale": 2.0,
                    }
                }
            },
        }
    )
    DummyRunner.results = [{"utt_id": "utt1", "posterior": np.array([1.0, 2.0])}]

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)

    inference_mod.infer(cfg)

    base = tmp_path / "infer" / "test_a"
    path = Path(_read_scp(base / "posterior.scp")[0].split(maxsplit=1)[1])
    assert path.suffixes[-2:] == [".custom", ".npy"]
    assert np.array_equal(np.load(path), np.array([2.0, 4.0]))


def test_inference_rejects_cuda_tensor_artifact(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in test env.")
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
    DummyRunner.results = [
        {"utt_id": "utt1", "posterior": torch.tensor([1.0], device="cuda")}
    ]

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)

    with pytest.raises(ValueError, match="Move the tensor to CPU"):
        inference_mod.infer(cfg)


def test_inference_rejects_top_level_list_outputs(tmp_path, monkeypatch):
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
    DummyRunner.results = [{"utt_id": "utt1", "audio": [1, 2, 3]}]

    monkeypatch.setattr(inference_mod, "set_parallel", lambda arg: None)

    with pytest.raises(TypeError, match="Top-level list outputs are not supported"):
        inference_mod.infer(cfg)


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
