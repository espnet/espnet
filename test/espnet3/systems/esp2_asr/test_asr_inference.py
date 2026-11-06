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
    cfg = OmegaConf.create(
        {
            "dataset": {
                "_target_": "espnet3.components.data.data_organizer.DataOrganizer",
                "test": [
                    {
                        "name": "dev",
                        "data_src": "mini_an4/esp2_asr",
                        "data_src_args": {"split": "dev"},
                    }
                ],
            },
            "test_set": "dev",
        }
    )
    calls = {}
    organizer = type("Org", (), {"test": {"dev": ["item"]}})()

    import espnet3.systems.base.inference_provider as provider_mod

    def fake_instantiate(obj):
        calls["obj"] = obj
        return organizer

    monkeypatch.setattr(provider_mod, "instantiate", fake_instantiate)

    dataset = InferenceProvider.build_dataset(cfg)

    assert dataset == ["item"]
    assert calls["obj"] == cfg.dataset


def test_build_model_cpu(monkeypatch):
    cfg = OmegaConf.create({"model": {"_target_": "dummy.Model"}})
    calls = {}

    import torch

    import espnet3.systems.base.inference_provider as provider_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

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

    import torch

    import espnet3.systems.base.inference_provider as provider_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

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

    import torch

    import espnet3.systems.base.inference_provider as provider_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

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

    import torch

    import espnet3.systems.base.inference_provider as provider_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

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


def test_forward_batch_failure_propagates():
    """A model that rejects a batch fails without per-item retry."""
    dataset = [
        {"utt_id": "utt1", "speech": "audio1", "text": "ref1"},
        {"utt_id": "utt2", "speech": "audio2", "text": "ref2"},
    ]

    class DummyModel:
        def __call__(self, speech):
            if isinstance(speech, list):
                raise TypeError("speech must be a single utterance")
            return [[f"hyp-{speech}"]]

    output_path = f"{__name__}._output_fn"
    with pytest.raises(TypeError, match="single utterance"):
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


class NativeASRProvider(InferenceProvider):
    @staticmethod
    def build_dataset(config):
        import numpy as np

        return [
            {"utt_id": f"u{i}", "speech": np.sin(np.arange(2000 + i * 160) / 20)}
            for i in range(3)
        ]


# Builds a real ASR model and decodes it; the first run pays the import cost,
# which exceeds the default 10-second execution timeout on CI.
@pytest.mark.execution_timeout(60)
@pytest.mark.parametrize("batch_size", [None, 2])
def test_native_asr_recipe_writes_the_backends_text(tmp_path, batch_size):
    """Exercise model construction, native decoding, formatting and SCP writing."""
    from espnet2.tasks.asr import ASRTask
    from espnet3.parallel.parallel import set_parallel

    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<blank>\n<unk>\na\nb\n<sos/eos>\n", encoding="utf-8")
    ASRTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "train"),
            "--token_list",
            str(tokens),
            "--token_type",
            "char",
            "--encoder",
            "transformer",
            "--encoder_conf",
            "{output_size: 8, attention_heads: 2, linear_units: 8, num_blocks: 1}",
            "--decoder",
            "transformer",
            "--decoder_conf",
            "{attention_heads: 2, linear_units: 8, num_blocks: 1}",
        ]
    )
    overrides = OmegaConf.create(
        {
            # This test loads TEMPLATE directly, bypassing launch's path context.
            "recipe_dir": str(tmp_path),
            "inference_dir": str(tmp_path / "decode"),
            "dataset": {"test": [{"name": "test"}]},
            "model": {
                "asr_train_config": str(tmp_path / "train" / "config.yaml"),
                "beam_size": 1,
                "ctc_weight": 1.0,
            },
            "batch_size": batch_size,
            "provider": {"_target_": f"{__name__}.NativeASRProvider"},
            "runner": {
                "_target_": "espnet3.systems.esp2_asr.inference.ASRInferenceRunner"
            },
            "parallel": {"env": "local", "n_workers": 1},
        }
    )
    from espnet3.utils.config_utils import load_default_config

    config = OmegaConf.merge(
        load_default_config("inference.yaml", "egs3.TEMPLATE.esp2_asr"), overrides
    )
    set_parallel(None)
    model = NativeASRProvider.build_model(config)
    expected = [
        f"{item['utt_id']} {model(speech=item['speech'])[0][0] or ''}"
        for item in NativeASRProvider.build_dataset(config)
    ]
    # Build another instance with identical initialized weights.
    checkpoint = tmp_path / "model.pth"
    import torch

    torch.save(model.asr_model.state_dict(), checkpoint)
    config.model.asr_model_file = str(checkpoint)
    # The publication API should decode the packed model through the same
    # inference config and output formatter as the recipe's infer stage.
    from espnet3.publication import InferenceModel

    pack = tmp_path / "model_pack"
    (pack / "conf").mkdir(parents=True)
    OmegaConf.save(config, pack / "conf" / "inference.yaml")
    OmegaConf.save(
        OmegaConf.create(
            {
                "schema_version": 1,
                "yaml_files": {"inference_config": "conf/inference.yaml"},
            }
        ),
        pack / "meta.yaml",
    )
    publication_model = InferenceModel.from_packed(pack, trust_user_code=True)
    from espnet3.publication.demo.assets import DEFAULT_UI_ASSETS
    from espnet3.publication.demo.session import DemoSession

    demo = DemoSession(
        pack,
        OmegaConf.create(
            {"model": {"call_args": {}}, "ui": {"title": None, "description": None}}
        ),
        publication_model,
        DEFAULT_UI_ASSETS.clone(),
    )
    callback = demo.create_inference_fn(demo.input_specs, demo.output_specs)
    samples = NativeASRProvider.build_dataset(config)
    batch_results = publication_model.forward_batch(samples)
    assert len(batch_results) == len(samples)
    assert [f"{item['utt_id']} {item['text']}" for item in batch_results] == expected
    reversed_results = publication_model.forward_batch(list(reversed(samples)))
    assert [item["text"] for item in reversed_results] == [
        item["text"] for item in reversed(batch_results)
    ]
    for item, expected_line in zip(NativeASRProvider.build_dataset(config), expected):
        result = publication_model(item, idx=item["utt_id"])
        assert isinstance(result, dict)
        assert f"{result['utt_id']} {result['text']}" == expected_line
        assert callback((16000, item["speech"])) == result["text"]
    try:
        infer(config)
        assert (
            tmp_path / "decode" / "test" / "text.scp"
        ).read_text().splitlines() == expected
    finally:
        set_parallel(None)

    from hydra.errors import InstantiationException

    checkpoint.unlink()
    with pytest.raises((FileNotFoundError, InstantiationException)) as caught:
        InferenceModel.from_packed(pack, trust_user_code=True)
    cause = caught.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(cause, FileNotFoundError)
    assert cause.filename == str(checkpoint)
