import pytest
from omegaconf import OmegaConf

from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import InferenceRunner


def _output_fn(*, data, model_output, idx):
    return {"uttid": data["uttid"], "hyp": model_output[0][0], "ref": data["text"]}


def test_build_dataset_uses_test_set(monkeypatch):
    cfg = OmegaConf.create({"dataset": {"_target_": "dummy"}, "test_set": "dev"})
    organizer = type("Org", (), {"test": {"dev": ["item"]}})()
    calls = {}

    def fake_instantiate(obj):
        calls["obj"] = obj
        return organizer

    monkeypatch.setattr(
        "espnet3.systems.base.inference_provider.instantiate", fake_instantiate
    )

    dataset = InferenceProvider.build_dataset(cfg)

    assert dataset == ["item"]
    assert calls["obj"] == cfg.dataset


def test_build_model_cpu(monkeypatch):
    cfg = OmegaConf.create({"model": {"_target_": "dummy.Model"}})
    calls = {}

    import espnet3.systems.base.inference_provider as provider_mod

    monkeypatch.setattr(provider_mod.torch.cuda, "is_available", lambda: False)

    def fake_instantiate(obj, device=None):
        calls["device"] = device
        calls["obj"] = obj
        return "model"

    monkeypatch.setattr(provider_mod, "instantiate", fake_instantiate)

    assert InferenceProvider.build_model(cfg) == "model"
    assert calls["device"] == "cpu"
    assert calls["obj"] == cfg.model


def test_build_model_cuda_uses_visible_device(monkeypatch):
    cfg = OmegaConf.create({"model": {"_target_": "dummy.Model"}})
    calls = {}

    import espnet3.systems.base.inference_provider as provider_mod

    monkeypatch.setattr(provider_mod.torch.cuda, "is_available", lambda: True)

    def fake_instantiate(obj, device=None):
        calls["device"] = device
        return "model"

    monkeypatch.setattr(provider_mod, "instantiate", fake_instantiate)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert InferenceProvider.build_model(cfg) == "model"
    assert calls["device"] == "cuda:2"


def test_forward_returns_hyp_and_ref():
    dataset = [{"uttid": "utt1", "speech": "audio", "text": "ref"}]

    class DummyModel:
        def __call__(self, speech):
            assert speech == "audio"
            return [["hyp"]]

    output_path = f"{__name__}._output_fn"
    out = InferenceRunner.forward(
        0,
        dataset=dataset,
        model=DummyModel(),
        input_key="speech",
        output_fn_path=output_path,
    )

    assert out == {"uttid": "utt1", "hyp": "hyp", "ref": "ref"}


def test_forward_requires_fields():
    dataset = [{"text": "ref"}]
    with pytest.raises(KeyError, match="Input key 'speech'"):
        InferenceRunner.forward(
            0,
            dataset=dataset,
            model=lambda x: [["hyp"]],
            input_key="speech",
            output_fn_path=f"{__name__}._output_fn",
        )

    dataset = [{"uttid": "utt1", "speech": "audio"}]
    with pytest.raises(KeyError, match="text"):
        InferenceRunner.forward(
            0,
            dataset=dataset,
            model=lambda x: [["hyp"]],
            input_key="speech",
            output_fn_path=f"{__name__}._output_fn",
        )
