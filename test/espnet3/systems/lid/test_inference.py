import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import espnet3.parallel.parallel as parallel_module
import espnet3.systems.lid.inference as inference_module
from egs3.voxlingua107.lid.src.inference import build_output
from espnet3.systems.base.inference import infer
from espnet3.systems.base.metric import measure
from espnet3.systems.lid.inference import Speech2Language


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


def test_voxlingua_output_formatter_handles_single_and_batch():
    assert build_output(
        {"lid_labels": "eng"},
        "eng",
        3,
    ) == {"utt_id": "3", "hyp": "eng", "ref": "eng"}
    assert build_output(
        [
            {"utt_id": "utt1", "lid_labels": "eng"},
            {"utt_id": "utt2", "lid_labels": "fra"},
        ],
        ["eng", "deu"],
        [0, 1],
    ) == [
        {"utt_id": "utt1", "hyp": "eng", "ref": "eng"},
        {"utt_id": "utt2", "hyp": "deu", "ref": "fra"},
    ]


@pytest.mark.parametrize("extract_embd", [False, True])
def test_lid_infer_and_measure_pipeline(tmp_path, monkeypatch, extract_embd):
    lang2utt = tmp_path / "lang2utt"
    lang2utt.write_text("eng 0\nfra 1\n", encoding="utf-8")
    monkeypatch.setattr(
        inference_module.LIDTask,
        "build_model_from_file",
        staticmethod(lambda **_kwargs: (DummyLIDModel(), None)),
    )
    monkeypatch.setattr(parallel_module, "parallel_config", None)

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
                "_target_": "espnet3.systems.lid.inference.Speech2Language",
                "lid_train_config": str(tmp_path / "config.yaml"),
                "lid_model_file": str(tmp_path / "model.pth"),
                "lang2utt": str(lang2utt),
                "extract_embd": extract_embd,
            },
            "input_key": "speech",
            "output_keys": (
                ["hyp", "ref", "embedding"] if extract_embd else ["hyp", "ref"]
            ),
            "idx_key": "utt_id",
            "batch_size": 2,
            "output_fn": "egs3.voxlingua107.lid.src.inference.build_output",
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
    assert (inference_dir / "dev/hyp.scp").read_text(encoding="utf-8") == (
        "0 eng\n1 fra\n"
    )
    assert (inference_dir / "dev/ref.scp").read_text(encoding="utf-8") == (
        "0 eng\n1 fra\n"
    )
    if extract_embd:
        rows = (inference_dir / "dev/embedding.scp").read_text().splitlines()
        assert len(rows) == 2
        for idx, row in enumerate(rows):
            utt_id, path = row.split(maxsplit=1)
            assert utt_id == str(idx)
            np.testing.assert_array_equal(np.load(path), np.zeros(2))
    else:
        assert not (inference_dir / "dev/embedding.scp").exists()

    results = measure(
        OmegaConf.create(
            {
                "inference_dir": str(inference_dir),
                "metrics": [
                    {
                        "metric": {
                            "_target_": (
                                "espnet3.systems.lid.metrics.accuracy.Accuracy"
                            )
                        }
                    }
                ],
            }
        )
    )
    metric_name = "espnet3.systems.lid.metrics.accuracy.Accuracy"
    assert results[metric_name]["dev"]["Accuracy"] == 100.0
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


def test_voxlingua_output_formatter_preserves_optional_embedding():
    embedding = np.array([0.6, 0.8], dtype=np.float32)
    record = build_output(
        {"lid_labels": "eng"}, {"hyp": "fra", "embedding": embedding}, 3
    )
    assert record["hyp"] == "fra"
    assert record["ref"] == "eng"
    assert record["embedding"] is embedding
