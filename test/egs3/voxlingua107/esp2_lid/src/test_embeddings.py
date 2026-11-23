"""Language embedding extraction, summaries and t-SNE plots."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from egs3.voxlingua107.esp2_lid.src.embeddings import (
    extract_embeddings,
    write_embeddings,
)


class FakeDataset:
    def __init__(self, languages):
        self.examples = [SimpleNamespace(language=language) for language in languages]
        self.read = []

    def __getitem__(self, idx):
        self.read.append(idx)
        return {
            "speech": np.full(2, idx, dtype=np.float32),
            "lid_labels": self.examples[idx].language,
        }


def fake_model(speech):
    """Return Speech2Language(extract_embd=True) outputs: the index as embedding."""
    return [{"hyp": "eng", "embedding": np.array([wave[0], 1.0])} for wave in speech]


def test_extract_keeps_the_first_utterances_of_each_language():
    dataset = FakeDataset(["eng", "eng", "fra", "eng", "fra"])

    languages = extract_embeddings(
        fake_model, dataset, max_utt_per_lang=2, batch_size=3
    )

    assert dataset.read == [0, 1, 2, 4]  # the third eng utterance is not read
    assert [embedding[0] for embedding in languages["eng"]] == [0, 1]
    assert [embedding[0] for embedding in languages["fra"]] == [2, 4]


def test_write_embeddings_and_normalized_averages(tmp_path):
    languages = {
        "eng": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        "fra": [np.array([0.0, 2.0])],
    }

    write_embeddings(languages, tmp_path, tsne=False)

    with np.load(tmp_path / "lang_to_embds.npz") as embeddings:
        np.testing.assert_array_equal(embeddings["eng"], [[1, 0], [0, 1]])
    with np.load(tmp_path / "lang_to_avg_embd.npz") as averages:
        np.testing.assert_allclose(averages["eng"], [2**-0.5, 2**-0.5])
        np.testing.assert_array_equal(averages["fra"], [0, 1])
    assert not (tmp_path / "tsne_plots").exists()


def test_tsne_reuses_espnet2_and_caps_perplexity(tmp_path, monkeypatch):
    calls = []
    module = types.ModuleType("espnet2.bin.lid_inference")
    module.gen_tsne_plot = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    languages = {
        language: [np.array(vector, dtype=np.float32)]
        for language, vector in (("eng", [1, 0]), ("fra", [0, 1]), ("deu", [-1, 0]))
    }

    write_embeddings(languages, tmp_path, seed=7)

    assert len(calls) == 2
    assert isinstance(calls[0][0][0]["eng"], list)
    assert isinstance(calls[1][0][0]["eng"], np.ndarray)
    assert calls[0][0][1] == str(tmp_path / "tsne_plots")
    assert calls[0][0][2] == 7
    assert calls[0][1] == calls[1][1] == {"perplexity": 2}


def test_single_embedding_is_not_plotted(tmp_path, monkeypatch):
    module = types.ModuleType("espnet2.bin.lid_inference")

    def unexpected(*args, **kwargs):
        pytest.fail("t-SNE must not run for a single point")

    module.gen_tsne_plot = unexpected
    monkeypatch.setitem(sys.modules, module.__name__, module)

    write_embeddings({"eng": [np.array([1.0, 0.0])]}, tmp_path)

    assert (tmp_path / "lang_to_embds.npz").is_file()
