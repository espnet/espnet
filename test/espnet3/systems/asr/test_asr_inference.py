import pytest
from omegaconf import OmegaConf

import espnet3.systems.asr.inference as asr_infer


def test_build_dataset_uses_test_set(monkeypatch):
    cfg = OmegaConf.create({"dataset": {"_target_": "dummy"}, "test_set": "dev"})
    organizer = type("Org", (), {"test": {"dev": ["item"]}})()
    calls = {}

    def fake_instantiate(obj):
        calls["obj"] = obj
        return organizer

    monkeypatch.setattr(asr_infer, "instantiate", fake_instantiate)

    dataset = asr_infer.InferenceProvider.build_dataset(cfg)

    assert dataset == ["item"]
    assert calls["obj"] == cfg.dataset


def test_build_model_cpu(monkeypatch):
    cfg = OmegaConf.create({"model": {"_target_": "dummy.Model"}})
    calls = {}

    monkeypatch.setattr(asr_infer.torch.cuda, "is_available", lambda: False)

    def fake_instantiate(obj, device=None):
        calls["device"] = device
        calls["obj"] = obj
        return "model"

    monkeypatch.setattr(asr_infer, "instantiate", fake_instantiate)

    assert asr_infer.InferenceProvider.build_model(cfg) == "model"
    assert calls["device"] == "cpu"
    assert calls["obj"] == cfg.model


def test_build_model_cuda_uses_visible_device(monkeypatch):
    cfg = OmegaConf.create({"model": {"_target_": "dummy.Model"}})
    calls = {}

    monkeypatch.setattr(asr_infer.torch.cuda, "is_available", lambda: True)

    def fake_instantiate(obj, device=None):
        calls["device"] = device
        return "model"

    monkeypatch.setattr(asr_infer, "instantiate", fake_instantiate)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert asr_infer.InferenceProvider.build_model(cfg) == "model"
    assert calls["device"] == "cuda:2"


def test_forward_returns_hyp_and_ref():
    dataset = [{"speech": "audio", "text": "ref"}]

    class DummyModel:
        def __call__(self, speech):
            assert speech == "audio"
            return [["hyp"]]

    out = asr_infer.InferenceRunner.forward(0, dataset=dataset, model=DummyModel())

    assert out == {"idx": 0, "hyp": "hyp", "ref": "ref"}


def test_forward_requires_fields():
    dataset = [{"text": "ref"}]
    with pytest.raises(AssertionError, match="requires 'speech'"):
        asr_infer.InferenceRunner.forward(0, dataset=dataset, model=lambda x: [["hyp"]])

    dataset = [{"speech": "audio"}]
    with pytest.raises(AssertionError, match="requires 'text'"):
        asr_infer.InferenceRunner.forward(0, dataset=dataset, model=lambda x: [["hyp"]])
