import sys
import types

import numpy as np
import pytest
from omegaconf import OmegaConf

from espnet3.systems.base.metric import measure
from espnet3.systems.lid.metrics.embedding import Embedding


def _write_inputs(tmp_path, rows):
    refs, paths = [], []
    for idx, (language, embedding) in enumerate(rows):
        path = tmp_path / f"{idx}.npy"
        np.save(path, np.array(embedding, dtype=np.float32))
        refs.append(f"{idx} {language}\n")
        paths.append(f"{idx} {path}\n")
    ref = tmp_path / "ref.scp"
    embedding = tmp_path / "embedding.scp"
    ref.write_text("".join(refs))
    embedding.write_text("".join(paths))
    return {"ref": ref, "embedding": embedding}


def test_embedding_summaries_and_limit(tmp_path):
    data = _write_inputs(
        tmp_path,
        [("eng", [1, 0]), ("eng", [0, 1]), ("eng", [-1, 0]), ("fra", [0, 1])],
    )
    result = Embedding(max_utt_per_lang=2)(data, "dev", tmp_path)
    assert result == {"Embeddings": 3, "Languages": 2}
    with np.load(tmp_path / "dev/dev_lang_to_embds.npz") as languages:
        np.testing.assert_array_equal(languages["eng"], [[1, 0], [0, 1]])
    with np.load(tmp_path / "dev/dev_lang_to_avg_embd.npz") as averages:
        np.testing.assert_allclose(averages["eng"], [2**-0.5, 2**-0.5])
        np.testing.assert_array_equal(averages["fra"], [0, 1])


def test_embedding_tsne_reuses_espnet2_and_caps_perplexity(tmp_path, monkeypatch):
    data = _write_inputs(tmp_path, [("eng", [1, 0]), ("fra", [0, 1]), ("deu", [-1, 0])])
    calls = []
    module = types.ModuleType("espnet2.bin.lid_inference")
    module.gen_tsne_plot = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    Embedding(save_tsne_plot=True, seed=7)(data, "dev", tmp_path)
    assert len(calls) == 2
    assert isinstance(calls[0][0][0]["eng"], list)
    assert isinstance(calls[1][0][0]["eng"], np.ndarray)
    assert calls[0][0][2] == 7
    assert calls[0][1] == {"perplexity": 2, "max_iter": 1000}
    assert calls[1][1] == {"perplexity": 2, "max_iter": 1000}


def test_embedding_singleton_does_not_run_tsne(tmp_path, monkeypatch):
    data = _write_inputs(tmp_path, [("eng", [1, 0])])
    module = types.ModuleType("espnet2.bin.lid_inference")

    def unexpected(*args, **kwargs):
        pytest.fail("t-SNE must not run for a single point")

    module.gen_tsne_plot = unexpected
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert Embedding(save_tsne_plot=True)(data, "dev", tmp_path)["Embeddings"] == 1


@pytest.mark.parametrize("embedding", [[float("nan"), 0], [[1, 2]], [1]])
def test_embedding_rejects_invalid_arrays(tmp_path, embedding):
    data = _write_inputs(tmp_path, [("eng", embedding)])
    with pytest.raises(ValueError, match="Invalid language embedding"):
        Embedding()(data, "dev", tmp_path)


def test_embedding_rejects_mismatched_scp_ids(tmp_path):
    data = _write_inputs(tmp_path, [("eng", [1, 0])])
    data["ref"].write_text("wrong eng\n")
    with pytest.raises(AssertionError, match="UID mismatch"):
        Embedding()(data, "dev", tmp_path)


def test_embedding_measure_pipeline(tmp_path):
    test_dir = tmp_path / "dev"
    test_dir.mkdir()
    _write_inputs(test_dir, [("eng", [1, 0]), ("fra", [0, 1])])
    metric = "espnet3.systems.lid.metrics.embedding.Embedding"
    config = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "metrics": [
                {
                    "metric": {"_target_": metric},
                    "inputs": {"ref": "ref", "embedding": "embedding"},
                }
            ],
        }
    )
    assert measure(config)[metric]["dev"] == {"Embeddings": 2, "Languages": 2}
