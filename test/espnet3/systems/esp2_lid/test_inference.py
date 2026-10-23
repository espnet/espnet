import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import espnet3.parallel.parallel as parallel_module
import espnet3.systems.base.inference as base_inference_module
import espnet3.systems.esp2_lid.src.speech2language as speech2language_module
from espnet3.contract.inference import load_inference_contract
from espnet3.contract.kinds.audio import Audio
from espnet3.publication import InferenceModel
from espnet3.systems.base.inference import infer
from espnet3.systems.base.metric import measure
from espnet3.systems.esp2_lid.src.inference_runner import ESP2LIDInferenceRunner
from espnet3.systems.esp2_lid.src.speech2language import Speech2Language
from espnet3.utils.config_utils import load_default_config


def _template():
    return load_default_config("inference.yaml", "egs3.TEMPLATE.esp2_lid")


class DummyLIDModel:
    def __init__(self):
        self.calls = []

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def __call__(self, speech, speech_lengths, **kwargs):
        self.calls.append(
            (speech.detach().cpu(), speech_lengths.detach().cpu(), kwargs)
        )
        predictions = torch.arange(speech.shape[0], device=speech.device)
        embeddings = torch.zeros(speech.shape[0], 2, device=speech.device)
        return embeddings, predictions


class DummyOrganizer:
    def __init__(self, test, **kwargs):
        del test, kwargs
        self.test = {
            "dev": [
                {
                    "speech": np.ones(160, dtype=np.float32),
                    "lid_labels": "eng",
                },
                {
                    "speech": np.ones(320, dtype=np.float32),
                    "lid_labels": "fra",
                },
            ]
        }


def test_speech2language_predicts_single_and_batched_inputs(tmp_path, monkeypatch):
    lang2utt = tmp_path / "lang2utt"
    lang2utt.write_text("eng 0\nfra 1\n", encoding="utf-8")
    model = DummyLIDModel()
    monkeypatch.setattr(
        speech2language_module.LIDTask,
        "build_model_from_file",
        staticmethod(lambda **_kwargs: (model, None)),
    )
    speech2language = Speech2Language(
        lid_train_config=tmp_path / "config.yaml",
        lid_model_file=tmp_path / "model.pth",
        lang2utt=lang2utt,
    )

    assert speech2language(np.ones(3, dtype=np.float32)) == "eng"
    assert speech2language(
        [np.ones(2, dtype=np.float32), np.ones(4, dtype=np.float32)]
    ) == ["eng", "fra"]

    padded, lengths, kwargs = model.calls[-1]
    assert padded.shape == (2, 4)
    assert lengths.tolist() == [2, 4]
    assert kwargs == {"lid_labels": None, "extract_embd": True}


def test_speech2language_rejects_non_waveform_input(tmp_path, monkeypatch):
    lang2utt = tmp_path / "lang2utt"
    lang2utt.write_text("eng 0\n", encoding="utf-8")
    monkeypatch.setattr(
        speech2language_module.LIDTask,
        "build_model_from_file",
        staticmethod(lambda **_kwargs: (DummyLIDModel(), None)),
    )
    speech2language = Speech2Language(
        lid_train_config=tmp_path / "config.yaml",
        lid_model_file=tmp_path / "model.pth",
        lang2utt=lang2utt,
    )

    with pytest.raises(ValueError, match="one-dimensional waveform"):
        speech2language(np.ones((2, 3), dtype=np.float32))


def test_speech2language_normalizes_optional_embeddings(tmp_path, monkeypatch):
    class Model(DummyLIDModel):
        def __call__(self, speech, speech_lengths, **kwargs):
            embeddings, predictions = super().__call__(speech, speech_lengths, **kwargs)
            embeddings[:] = torch.tensor([3.0, 4.0])
            return embeddings, predictions

    lang2utt = tmp_path / "lang2utt"
    lang2utt.write_text("eng 0\nfra 1\n", encoding="utf-8")
    monkeypatch.setattr(
        speech2language_module.LIDTask,
        "build_model_from_file",
        staticmethod(lambda **_kwargs: (Model(), None)),
    )
    model = Speech2Language("config.yaml", "model.pth", lang2utt, extract_embd=True)
    prediction = model(np.ones(3, dtype=np.float32))
    assert prediction["hyp"] == "eng"
    np.testing.assert_allclose(prediction["embedding"], [0.6, 0.8])
    predictions = model([np.ones(3), np.ones(4)])
    assert [record["hyp"] for record in predictions] == ["eng", "fra"]
    for prediction in predictions:
        np.testing.assert_allclose(np.linalg.norm(prediction["embedding"]), 1.0)


class FakeSpeech2Language:
    """Return what Speech2Language does, remembering its input."""

    def __init__(self, extract_embd=False):
        self.extract_embd = extract_embd
        self.seen = None

    def __call__(self, speech):
        self.seen = speech
        if isinstance(speech, list):
            return [self._predict(index) for index in range(len(speech))]
        return self._predict(0)

    def _predict(self, index):
        language = ["jpn", "eng"][index % 2]
        if self.extract_embd:
            return {"hyp": language, "embedding": np.zeros(2, dtype=np.float32)}
        return language


def test_runner_formats_the_language_of_strings_and_embedding_dicts():
    assert ESP2LIDInferenceRunner.format_output("jpn", batched=False) == {
        "language": "jpn"
    }
    assert ESP2LIDInferenceRunner.format_output(
        ["jpn", {"hyp": "eng", "embedding": np.zeros(2)}], batched=True
    ) == [{"language": "jpn"}, {"language": "eng"}]


def test_runner_passes_mono_samples_to_the_backend():
    audio = Audio(np.zeros(160, dtype=np.float32), 16000)
    single = ESP2LIDInferenceRunner.prepare_model_inputs({"speech": audio})
    batch = ESP2LIDInferenceRunner.prepare_model_inputs({"speech": [audio, audio]})
    assert single["speech"].shape == (160,)
    assert [wave.shape for wave in batch["speech"]] == [(160,), (160,)]


@pytest.mark.parametrize("extract_embd", [False, True])
def test_template_contract_returns_only_the_language(extract_embd):
    """The TEMPLATE contract checks the input and the language-only output."""
    config = _template()
    contract = load_inference_contract(config.contract)
    dataset = [
        {"speech": np.zeros(160, dtype=np.float32), "lid_labels": "jpn"},
        {"speech": np.zeros((2, 320), dtype=np.float32), "lid_labels": "eng"},
    ]
    backend = FakeSpeech2Language(extract_embd=extract_embd)
    kwargs = dict(
        dataset=dataset,
        model=backend,
        input_key=config.input_key,
        inference_contract=contract,
    )

    assert ESP2LIDInferenceRunner.forward(0, **kwargs) == {
        "language": "jpn",
        "utt_id": "0",
    }
    assert ESP2LIDInferenceRunner.forward([0, 1], **kwargs) == [
        {"language": "jpn", "utt_id": "0"},
        {"language": "eng", "utt_id": "1"},
    ]
    # first_channel turns the stereo item into the mono waveform the backend takes.
    assert [wave.shape for wave in backend.seen] == [(160,), (320,)]


@pytest.mark.parametrize("extract_embd", [False, True])
def test_lid_infer_and_measure_pipeline(tmp_path, monkeypatch, extract_embd):
    """The recipe's stages: infer writes language.scp, measure reads lid_labels."""
    lang2utt = tmp_path / "lang2utt"
    lang2utt.write_text("eng 0\nfra 1\n", encoding="utf-8")
    monkeypatch.setattr(
        speech2language_module.LIDTask,
        "build_model_from_file",
        staticmethod(lambda **_kwargs: (DummyLIDModel(), None)),
    )
    monkeypatch.setattr(parallel_module, "parallel_config", None)
    # Run shards in this process so the patched model builder is used.
    monkeypatch.setattr(base_inference_module, "set_parallel", lambda config: None)

    inference_dir = tmp_path / "inference"
    inference_config = OmegaConf.merge(
        _template(),
        {
            "inference_dir": str(inference_dir),
            "dataset": {
                "_target_": f"{__name__}.DummyOrganizer",
                "_recursive_": False,
                "test": [{"name": "dev"}],
            },
            "model": {
                "lid_train_config": str(tmp_path / "config.yaml"),
                "lid_model_file": str(tmp_path / "model.pth"),
                "lang2utt": str(lang2utt),
                "extract_embd": extract_embd,
            },
        },
    )

    infer(inference_config)
    assert sorted(path.name for path in (inference_dir / "dev").glob("*.scp")) == [
        "language.scp"
    ]
    assert (inference_dir / "dev/language.scp").read_text(encoding="utf-8") == (
        "0 eng\n1 fra\n"
    )

    metrics_config = OmegaConf.merge(
        load_default_config("metrics.yaml", "egs3.TEMPLATE.esp2_lid"),
        {"inference_dir": str(inference_dir), "exp_dir": str(tmp_path)},
    )
    results = measure(metrics_config, inference_config=inference_config)
    prefix = "espnet3.systems.esp2_lid.metrics"
    assert results[f"{prefix}.accuracy.Accuracy"]["dev"] == {
        "MicroAccuracy": 100.0,
        "MacroAccuracy": 100.0,
    }
    assert (inference_dir / "dev/dataset/lid_labels.scp").read_text() == (
        "0 eng\n1 fra\n"
    )
    assert (inference_dir / "dev/lid_per_language.json").is_file()
    assert (inference_dir / "metrics.json").is_file()


@pytest.mark.parametrize("extract_embd", [False, True])
def test_inference_model_serves_a_bundle_with_the_template_config(
    tmp_path, monkeypatch, extract_embd
):
    """A packed LID config runs through the same runner and contract."""
    monkeypatch.setattr(
        speech2language_module.LIDTask,
        "build_model_from_file",
        staticmethod(lambda **_kwargs: (DummyLIDModel(), None)),
    )
    pack = tmp_path / "model_pack"
    (pack / "conf").mkdir(parents=True)
    (pack / "exp/stats/train").mkdir(parents=True)
    (pack / "exp/stats/train/lang2utt").write_text("jpn 0\neng 1\n")
    config = OmegaConf.merge(
        _template(),
        {
            "recipe_dir": ".",
            "exp_tag": "training",
            "model": {
                "lid_train_config": "${recipe_dir}/exp/config.yaml",
                "lid_model_file": "${recipe_dir}/exp/model.pth",
                "lang2utt": "${recipe_dir}/exp/stats/train/lang2utt",
                "extract_embd": extract_embd,
            },
        },
    )
    OmegaConf.save(config, pack / "conf" / "inference.yaml")
    OmegaConf.save(
        OmegaConf.create(
            {
                "schema_version": 1,
                "system": "esp2_lid",
                "yaml_files": {"inference_config": "conf/inference.yaml"},
            }
        ),
        pack / "meta.yaml",
    )

    model = InferenceModel.from_packed(pack, device="cpu")
    result = model(np.zeros(160, dtype=np.float32))
    batch = model.forward_batch([np.zeros(160), np.zeros(320)])

    assert result["language"] == "jpn"
    assert "embedding" not in result
    assert [item["language"] for item in batch] == ["jpn", "eng"]


def test_pack_model_bundle_with_the_template_configs_loads(tmp_path, monkeypatch):
    """pack_model with the TEMPLATE configs bundles lang2utt for InferenceModel."""
    from espnet3.utils.publication_utils import pack_model

    monkeypatch.setattr(
        speech2language_module.LIDTask,
        "build_model_from_file",
        staticmethod(lambda **_kwargs: (DummyLIDModel(), None)),
    )
    recipe = tmp_path / "recipe"
    exp_dir = recipe / "exp" / "training"
    exp_dir.mkdir(parents=True)
    (exp_dir / "config.yaml").write_text("dummy: true\n")
    (exp_dir / "valid.accuracy.best.pth").write_bytes(b"weights")
    stats_dir = recipe / "exp" / "stats"
    (stats_dir / "train").mkdir(parents=True)
    (stats_dir / "train" / "lang2utt").write_text("jpn 0\neng 1\n")

    training = OmegaConf.create(
        {
            "task": "espnet3.systems.esp2_lid.task.LIDTask",
            "recipe_dir": str(recipe),
            "exp_dir": str(exp_dir),
        }
    )
    inference = OmegaConf.merge(
        _template(),
        {
            "recipe_dir": str(recipe),
            "exp_dir": str(exp_dir),
            "inference_dir": str(exp_dir / "inference"),
            "stats_dir": str(stats_dir),
            "model": {
                "lid_train_config": "${exp_dir}/config.yaml",
                "lid_model_file": "${exp_dir}/valid.accuracy.best.pth",
                "lang2utt": "${stats_dir}/train/lang2utt",
            },
        },
    )
    OmegaConf.resolve(inference)
    publication = OmegaConf.merge(
        load_default_config("publication.yaml", "egs3.TEMPLATE.esp2_lid"),
        {
            "recipe_dir": str(recipe),
            "exp_tag": "training",
            "pack_model": {"out_dir": str(tmp_path / "pack"), "readme": None},
            "upload_model": {"hf_repo": "user/model"},
        },
    )
    OmegaConf.resolve(publication)

    pack = pack_model(
        training_config=training,
        publication_config=publication,
        inference_config=inference,
    )

    assert (pack / "exp" / "stats" / "train" / "lang2utt").is_file()
    model = InferenceModel.from_packed(pack, device="cpu")
    assert model(np.zeros(160, dtype=np.float32))["language"] == "jpn"
