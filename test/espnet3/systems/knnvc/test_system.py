"""Tests for the ESPnet3 VC system stage hooks."""

from pathlib import Path
from test.espnet3.systems.knnvc import fixtures

import numpy as np
import pytest
from omegaconf import OmegaConf

from espnet3.systems.knnvc.system import KNNVCSystem

# ===============================================================
# Test Case Summary
# ===============================================================
#
# prepare_features
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_prepare_features_end_to_end            | Writes one npy per utterance |
# |                          | and one feats.<name>.scp per dataset entry.     |
# | test_prepare_features_requires_config       | Missing config sections      |
# |                                             | raise RuntimeError.          |
# | test_prepare_features_rejects_stage_args    | Stage arguments raise        |
# |                                             | TypeError.                   |
# | test_stage_log_dir_mapping                  | prepare_features logs go to  |
# |                                             | features_dir.                |
# | test_indices_are_grouped_by_pool            | Utterances reach the runner  |
# |                          | one pool at a time even when the dataset order   |
# |                          | interleaves pools.                               |

FIXTURES = fixtures.__name__


def _training_config(tmp_path, **prepare_overrides):
    prepare_features = {
        "features_dir": str(tmp_path / "features"),
        "dataset": [
            {
                "name": "train",
                "data_src": FIXTURES,
                "data_src_args": {"split": "train"},
            },
            {"name": "dev", "data_src": FIXTURES, "data_src_args": {"split": "dev"}},
        ],
        "encoder": {"_target_": f"{FIXTURES}.DummyEncoder"},
        "prematch": True,
        "topk": 2,
        "device": "cpu",
    }
    prepare_features.update(prepare_overrides)
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "recipe_dir": str(tmp_path),
            "parallel": {"env": "local", "n_workers": 1},
            "prepare_features": prepare_features,
        }
    )


def test_prepare_features_end_to_end(tmp_path):
    system = KNNVCSystem(training_config=_training_config(tmp_path))
    system.prepare_features()

    features_dir = tmp_path / "features"
    for name in ("train", "dev"):
        scp = features_dir / f"feats.{name}.scp"
        assert scp.is_file()
        lines = scp.read_text().strip().splitlines()
        assert len(lines) == len(fixtures.UTTERANCES)
        for line in lines:
            feature_name, path = line.split(maxsplit=1)
            assert feature_name.startswith(f"{name}/")
            feats = np.load(path)
            assert feats.dtype == np.float16
            assert feats.shape[1] == fixtures.FEATURE_DIM
        assert (features_dir / "shards" / name / "split.0" / "done").exists()


@pytest.mark.parametrize("missing", ["features_dir", "dataset", "encoder"])
def test_prepare_features_requires_config(tmp_path, missing):
    config = _training_config(tmp_path)
    config.prepare_features.pop(missing)
    system = KNNVCSystem(training_config=config)
    with pytest.raises(RuntimeError, match=missing):
        system.prepare_features()

    system = KNNVCSystem(training_config=OmegaConf.create({"exp_dir": str(tmp_path)}))
    with pytest.raises(RuntimeError, match="prepare_features must be set"):
        system.prepare_features()


def test_prepare_features_rejects_stage_args(tmp_path):
    system = KNNVCSystem(training_config=_training_config(tmp_path))
    with pytest.raises(TypeError, match="does not accept arguments"):
        system.prepare_features("unexpected")


def test_stage_log_dir_mapping(tmp_path):
    system = KNNVCSystem(training_config=_training_config(tmp_path))
    assert system.stage_log_dirs["prepare_features"] == Path(tmp_path / "features")
    assert system.stage_log_dirs["train"] == Path(tmp_path / "exp")


def test_indices_are_grouped_by_pool(tmp_path, monkeypatch):
    """The runner must receive one pool's utterances at a time.

    ``PrepareFeaturesRunner`` caches the encoded features of a single pool, so
    interleaved indices would re-encode each pool once per utterance. The
    fixture deliberately interleaves pools in dataset order, so index order
    alone is not enough.
    """
    from espnet3.systems.knnvc import system as system_module

    seen = []
    original = system_module.PrepareFeaturesRunner.__call__

    def _record(self, indices, *args, **kwargs):
        seen.append(list(indices))
        return original(self, indices, *args, **kwargs)

    monkeypatch.setattr(system_module.PrepareFeaturesRunner, "__call__", _record)

    config = _training_config(tmp_path)
    for entry in config.prepare_features.dataset:
        entry.data_src_args.interleaved = True
    KNNVCSystem(training_config=config).prepare_features()

    assert seen, "the runner was never called"
    for indices in seen:
        keys = [fixtures.INTERLEAVED_UTTERANCES[i][1] for i in indices]
        # Every pool occupies one contiguous run of the index list.
        runs = [key for pos, key in enumerate(keys) if pos == 0 or key != keys[pos - 1]]
        assert len(runs) == len(set(runs)), f"pools are interleaved: {keys}"
