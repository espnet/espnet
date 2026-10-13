from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf

import espnet3.parallel.parallel as parallel_module
import espnet3.systems.base.inference as base_inference_module
import espnet3.systems.esp2_lid.inference as inference_module
from espnet3.api.inference import Audio, load
from espnet3.systems.base.inference import infer
from espnet3.systems.base.inference_runner import InferenceRunner
from espnet3.systems.base.metric import measure
from espnet3.systems.esp2_lid.inference import Inference, Speech2Language


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
    def __init__(self, test):
        del test
        self.test = {
            "dev": [
                {
                    "speech": np.ones(2, dtype=np.float32),
                    "lid_labels": "eng",
                },
                {
                    "speech": np.ones(4, dtype=np.float32),
                    "lid_labels": "fra",
                },
            ]
        }


def test_speech2language_predicts_single_and_batched_inputs(tmp_path, monkeypatch):
    lang2utt = tmp_path / "lang2utt"
    lang2utt.write_text("eng 0\nfra 1\n", encoding="utf-8")
    model = DummyLIDModel()
    monkeypatch.setattr(
        inference_module.LIDTask,
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
        inference_module.LIDTask,
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


@pytest.mark.parametrize("extract_embd", [False, True])
def test_lid_infer_and_measure_pipeline(tmp_path, monkeypatch, extract_embd):
    """The recipe's stages: infer writes language.scp, measure reads lid_labels."""
    lang2utt = tmp_path / "lang2utt"
    lang2utt.write_text("eng 0\nfra 1\n", encoding="utf-8")
    monkeypatch.setattr(
        inference_module.LIDTask,
        "build_model_from_file",
        staticmethod(
            lambda **_kwargs: (DummyLIDModel(), SimpleNamespace(frontend_conf={}))
        ),
    )
    monkeypatch.setattr(parallel_module, "parallel_config", None)
    # Run shards in this process so the patched model builder is used.
    monkeypatch.setattr(base_inference_module, "set_parallel", lambda config: None)

    inference_dir = tmp_path / "inference"
    inference_config = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "parallel": {"env": "local", "n_workers": 1},
            "dataset": {
                "_target_": f"{__name__}.DummyOrganizer",
                "_recursive_": False,
                "test": [{"name": "dev"}],
            },
            "model": {
                "_target_": "espnet3.systems.esp2_lid.inference.Inference",
                "lid_train_config": str(tmp_path / "config.yaml"),
                "lid_model_file": str(tmp_path / "model.pth"),
                "lang2utt": str(lang2utt),
                "extract_embd": extract_embd,
            },
            "batch_size": 2,
            "provider": {
                "_target_": (
                    "espnet3.systems.base.inference_provider.InferenceProvider"
                )
            },
            "runner": {
                "_target_": "espnet3.systems.base.inference_runner.InferenceRunner"
            },
        }
    )

    infer(inference_config)
    assert sorted(path.name for path in (inference_dir / "dev").glob("*.scp")) == [
        "language.scp"
    ]
    assert (inference_dir / "dev/language.scp").read_text(encoding="utf-8") == (
        "0 eng\n1 fra\n"
    )

    prefix = "espnet3.systems.esp2_lid.metrics"
    inputs = {"ref": "dataset:lid_labels", "hyp": "language"}
    results = measure(
        OmegaConf.create(
            {
                "inference_dir": str(inference_dir),
                "metrics": [
                    {
                        "metric": {"_target_": f"{prefix}.accuracy.Accuracy"},
                        "inputs": inputs,
                    },
                    {
                        "metric": {
                            "_target_": (
                                f"{prefix}.classification_report.ClassificationReport"
                            )
                        },
                        "inputs": inputs,
                    },
                ],
            }
        ),
        inference_config=inference_config,
    )
    assert results[f"{prefix}.accuracy.Accuracy"]["dev"] == {
        "MicroAccuracy": 100.0,
        "MacroAccuracy": 100.0,
    }
    assert (inference_dir / "dev/dataset/lid_labels.scp").read_text() == (
        "0 eng\n1 fra\n"
    )
    assert (inference_dir / "dev/lid_per_language.json").is_file()
    assert (inference_dir / "metrics.json").is_file()


def test_speech2language_normalizes_optional_embeddings(tmp_path, monkeypatch):
    class Model(DummyLIDModel):
        def __call__(self, speech, speech_lengths, **kwargs):
            embeddings, predictions = super().__call__(speech, speech_lengths, **kwargs)
            embeddings[:] = torch.tensor([3.0, 4.0])
            return embeddings, predictions

    lang2utt = tmp_path / "lang2utt"
    lang2utt.write_text("eng 0\nfra 1\n", encoding="utf-8")
    monkeypatch.setattr(
        inference_module.LIDTask,
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

    def __init__(self, fs=None, extract_embd=False, **kwargs):
        self.lid_train_args = SimpleNamespace(
            frontend_conf={} if fs is None else {"fs": fs}
        )
        self.extract_embd = extract_embd
        self.kwargs = kwargs
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


def test_inference_returns_the_predicted_language():
    backend = FakeSpeech2Language()
    assert Inference(backend)(np.zeros(16000, dtype=np.float32)) == {"language": "jpn"}
    assert backend.seen.ndim == 1 and backend.seen.dtype == np.float32


def test_inference_reads_the_language_from_an_embedding_backend():
    model = Inference(FakeSpeech2Language(extract_embd=True))
    assert model(np.zeros(160, dtype=np.float32)) == {"language": "jpn"}
    assert model.batch([{"speech": np.zeros(160)}, {"speech": np.zeros(320)}]) == [
        {"language": "jpn"},
        {"language": "eng"},
    ]


def test_inference_batch_is_one_backend_call():
    backend = FakeSpeech2Language()
    Inference(backend).batch([{"speech": np.zeros(160)}, {"speech": np.zeros(320)}])
    assert [len(wave) for wave in backend.seen] == [160, 320]


def test_inference_sample_rate_comes_from_the_frontend_config():
    assert Inference(FakeSpeech2Language()).sample_rate == 16000
    assert Inference(FakeSpeech2Language(fs="16k")).sample_rate == 16000
    backend = FakeSpeech2Language(fs=8000)
    Inference(backend)(Audio(np.zeros(16000, dtype=np.float32), 16000))
    assert len(backend.seen) == 8000
    with pytest.raises(TypeError, match="cannot tell the rate"):
        Inference(SimpleNamespace()).sample_rate  # not a Speech2Language at all


def test_the_infer_stage_runner_calls_inference_as_it_is():
    """InferenceRunner needs no output_fn: the declared output is written."""
    model = Inference(FakeSpeech2Language())
    dataset = {
        i: {"speech": np.zeros(160, dtype=np.float32), "lid_labels": "jpn"}
        for i in range(3)
    }
    assert InferenceRunner.forward(0, dataset=dataset, model=model) == {
        "utt_id": "0",
        "language": "jpn",
    }
    assert InferenceRunner.forward([1, 2], dataset=dataset, model=model) == [
        {"utt_id": "1", "language": "jpn"},
        {"utt_id": "2", "language": "eng"},
    ]


def test_speech2language_keeps_its_training_args(tmp_path, monkeypatch):
    lang2utt = tmp_path / "lang2utt"
    lang2utt.write_text("eng 0\n", encoding="utf-8")
    args = SimpleNamespace(frontend_conf={"fs": 16000})
    monkeypatch.setattr(
        inference_module.LIDTask,
        "build_model_from_file",
        staticmethod(lambda **_kwargs: (DummyLIDModel(), args)),
    )
    speech2language = Speech2Language("config.yaml", "model.pth", lang2utt)
    assert speech2language.lid_train_args is args


@pytest.mark.parametrize(
    "packed_model, packed_system",
    [
        ("Inference", "esp2_lid"),
        # The published VoxLingua107 bundle: packed before the recipe named
        # Inference as its model and before meta.yaml recorded the system.
        ("Speech2Language", None),
    ],
)
def test_load_serves_a_pack_made_by_the_recipe(
    tmp_path, monkeypatch, packed_model, packed_system
):
    """A bundle naming Inference, or Speech2Language as older recipes packed."""
    # The real Speech2Language, around a stand-in model: a bundle may only build
    # espnet2/espnet3 classes, so the class itself is not replaced.
    monkeypatch.setattr(
        inference_module.LIDTask,
        "build_model_from_file",
        staticmethod(
            lambda **_kwargs: (DummyLIDModel(), SimpleNamespace(frontend_conf={}))
        ),
    )
    lang2utt = tmp_path / "lang2utt"
    lang2utt.write_text("jpn 0\neng 1\n", encoding="utf-8")
    pack = tmp_path / "pack"
    (pack / "conf").mkdir(parents=True)
    meta = {
        "schema_version": 1,
        "yaml_files": {"inference_config": "conf/inference.yaml"},
    }
    if packed_system:
        meta["system"] = packed_system
    (pack / "meta.yaml").write_text(yaml.safe_dump(meta))
    inference_config = {
        "model": {
            "_target_": f"espnet3.systems.esp2_lid.inference.{packed_model}",
            "lid_train_config": str(tmp_path / "config.yaml"),
            "lid_model_file": str(tmp_path / "model.pth"),
            "lang2utt": str(lang2utt),
            "extract_embd": True,
        },
    }
    if packed_model == "Speech2Language":
        inference_config.update(
            input_key="speech", output_fn="src.inference.build_output"
        )
    (pack / "conf" / "inference.yaml").write_text(yaml.safe_dump(inference_config))

    model = load(pack) if packed_system else load(pack, system="esp2_lid")

    assert isinstance(model, Inference)
    assert isinstance(model.backend, Speech2Language)
    assert model.sample_rate == 16000
    assert model(np.zeros(160, dtype=np.float32)) == {"language": "jpn"}
