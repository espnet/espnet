import pytest
from omegaconf import OmegaConf

from espnet3.systems.base.inference import infer
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import InferenceRunner


def _output_fn(*, data, model_output, idx):
    return {"utt_id": data["utt_id"], "hyp": model_output[0][0], "ref": data["text"]}


def _batch_output_fn(*, data, model_output, idx):
    return [
        {
            "utt_id": sample["utt_id"],
            "hyp": output[0][0],
            "ref": sample["text"],
        }
        for sample, output in zip(data, model_output)
    ]


def _param_output_fn(*, data, model_output, idx):
    return {"utt_id": data["utt_id"], "hyp": model_output, "ref": "ref"}


class DummyProvider(InferenceProvider):
    @staticmethod
    def build_dataset(config):
        return config.dataset.data

    @staticmethod
    def build_model(config):
        def model(speech):
            return f"base-{speech}"

        return model


class ParamRunner(InferenceRunner):
    @staticmethod
    def forward(idx, *, dataset=None, model=None, flip=False, **kwargs):
        output = InferenceRunner.forward(idx, dataset=dataset, model=model, **kwargs)
        output["hyp"] = "flip" if flip else "base"
        return output


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


def test_build_model_cuda_defaults_to_logical_device_zero(monkeypatch):
    cfg = OmegaConf.create({"model": {"_target_": "dummy.Model"}})
    calls = {}

    import espnet3.systems.base.inference_provider as provider_mod

    monkeypatch.setattr(provider_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(provider_mod.torch.cuda, "device_count", lambda: 2)

    def fake_instantiate(obj, device=None):
        calls["device"] = device
        return "model"

    monkeypatch.setattr(provider_mod, "instantiate", fake_instantiate)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert InferenceProvider.build_model(cfg) == "model"
    assert calls["device"] == "cuda:0"


def test_build_model_cuda_uses_logical_device_index(monkeypatch):
    cfg = OmegaConf.create({"model": {"_target_": "dummy.Model"}, "device_index": 1})
    calls = {}

    import espnet3.systems.base.inference_provider as provider_mod

    monkeypatch.setattr(provider_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(provider_mod.torch.cuda, "device_count", lambda: 2)

    def fake_instantiate(obj, device=None):
        calls["device"] = device
        return "model"

    monkeypatch.setattr(provider_mod, "instantiate", fake_instantiate)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert InferenceProvider.build_model(cfg) == "model"
    assert calls["device"] == "cuda:1"


def test_build_model_cuda_respects_explicit_device(monkeypatch):
    cfg = OmegaConf.create({"model": {"_target_": "dummy.Model"}, "device": "cuda:7"})
    calls = {}

    import espnet3.systems.base.inference_provider as provider_mod

    monkeypatch.setattr(provider_mod.torch.cuda, "is_available", lambda: True)

    def fake_instantiate(obj, device=None):
        calls["device"] = device
        return "model"

    monkeypatch.setattr(provider_mod, "instantiate", fake_instantiate)

    assert InferenceProvider.build_model(cfg) == "model"
    assert calls["device"] == "cuda:7"


def test_forward_returns_hyp_and_ref():
    dataset = [{"utt_id": "utt1", "speech": "audio", "text": "ref"}]

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

    assert out == {"utt_id": "utt1", "hyp": "hyp", "ref": "ref"}


def test_forward_without_output_fn_returns_raw_model_output():
    dataset = [{"utt_id": "utt1", "speech": "audio", "text": "ref"}]

    class DummyModel:
        def __call__(self, speech):
            assert speech == "audio"
            return {"utt_id": "utt1", "hyp": "hyp"}

    out = InferenceRunner.forward(
        0,
        dataset=dataset,
        model=DummyModel(),
        input_key="speech",
    )

    assert out == {"utt_id": "utt1", "hyp": "hyp"}


def test_forward_batch_with_batched_inputs():
    dataset = [
        {"utt_id": "utt1", "speech": "audio1", "text": "ref1"},
        {"utt_id": "utt2", "speech": "audio2", "text": "ref2"},
    ]

    class DummyModel:
        def __call__(self, **inputs):
            assert inputs == {"speech": ["audio1", "audio2"]}
            return [[["hyp1"]], [["hyp2"]]]

    output_path = f"{__name__}._batch_output_fn"
    out = InferenceRunner.forward(
        [0, 1],
        dataset=dataset,
        model=DummyModel(),
        input_key="speech",
        output_fn_path=output_path,
    )

    assert out == [
        {"utt_id": "utt1", "hyp": "hyp1", "ref": "ref1"},
        {"utt_id": "utt2", "hyp": "hyp2", "ref": "ref2"},
    ]


def test_forward_batch_requires_batched_model():
    dataset = [
        {"utt_id": "utt1", "speech": "audio1", "text": "ref1"},
        {"utt_id": "utt2", "speech": "audio2", "text": "ref2"},
    ]

    class DummyModel:
        def __call__(self, speech):
            return [[f"hyp-{speech}"]]

    output_path = f"{__name__}._output_fn"
    with pytest.raises(RuntimeError, match="Batched inference failed"):
        InferenceRunner.forward(
            [0, 1],
            dataset=dataset,
            model=DummyModel(),
            input_key="speech",
            output_fn_path=output_path,
        )


def test_forward_requires_fields():
    dataset = [{"text": "ref"}]
    with pytest.raises(KeyError, match="Input key 'speech'"):
        InferenceRunner.forward(
            0,
            dataset=dataset,
            model=lambda **kwargs: [["hyp"]],
            input_key="speech",
            output_fn_path=f"{__name__}._output_fn",
        )

    dataset = [{"utt_id": "utt1", "speech": "audio"}]
    with pytest.raises(KeyError, match="text"):
        InferenceRunner.forward(
            0,
            dataset=dataset,
            model=lambda **kwargs: [["hyp"]],
            input_key="speech",
            output_fn_path=f"{__name__}._output_fn",
        )


def test_inference_requires_provider_config():
    cfg = OmegaConf.create(
        {
            "inference_dir": "unused",
            "dataset": {"test": [{"name": "test"}]},
            "input_key": "speech",
            "output_fn": f"{__name__}._output_fn",
            "runner": {"_target_": f"{__name__}.ParamRunner"},
            "parallel": {"env": "local", "n_workers": 1},
        }
    )
    with pytest.raises(RuntimeError, match="inference_config.provider must be set"):
        infer(cfg)


@pytest.mark.parametrize("flip,expected", [(False, "base"), (True, "flip")])
def test_inference_params_affect_runner_forward(tmp_path, flip, expected):
    cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {
                "test": [{"name": "test"}],
                "data": [{"utt_id": "utt1", "speech": "s1"}],
            },
            "input_key": "speech",
            "output_fn": f"{__name__}._param_output_fn",
            "provider": {
                "_target_": f"{__name__}.DummyProvider",
                "params": {"flip": flip},
            },
            "runner": {"_target_": f"{__name__}.ParamRunner"},
            "parallel": {"env": "local", "n_workers": 1},
        }
    )

    infer(cfg)

    scp_path = tmp_path / "test" / "hyp.scp"
    assert scp_path.exists()
    assert scp_path.read_text().strip() == f"utt1 {expected}"
