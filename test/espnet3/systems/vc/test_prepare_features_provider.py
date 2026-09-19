"""Tests for the prepare_features environment provider."""

from test.espnet3.systems.vc import fixtures

import pytest
from omegaconf import OmegaConf

from espnet3.parallel.parallel import set_parallel
from espnet3.systems.vc.prepare_features_provider import PrepareFeaturesProvider
from espnet3.systems.vc.prepare_features_runner import PoolFeatureCache

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_provider_requires_params               | Missing dataset/encoder/     |
# |                                             | features_dir raise.          |
# | test_provider_rejects_dataset_without_contract | Dataset lacking the stage |
# |                                             | methods raises TypeError.    |
# | test_build_env_local_contents               | env has dataset, encoder,    |
# |                                             | pool map and options.        |
# | test_build_worker_setup_fn_matches_local    | The worker setup fn builds   |
# |                                             | the same pool map/options.   |
# | test_build_env_local_reuses_driver_dataset  | The corpus is indexed once   |
# |                                             | on the driver.               |


FIXTURES = fixtures.__name__


def _params(features_dir, **overrides):
    params = {
        "dataset": {"data_src": FIXTURES, "data_src_args": {"split": "train"}},
        "encoder": {"_target_": f"{FIXTURES}.DummyEncoder"},
        "features_dir": str(features_dir),
        "prematch": True,
        "topk": 2,
        "device": "cpu",
    }
    params.update(overrides)
    return params


@pytest.fixture(autouse=True)
def _local_parallel():
    set_parallel(OmegaConf.create({"env": "local", "n_workers": 1}))


# ---------------------------------------------------------------
# PrepareFeaturesProvider
# ---------------------------------------------------------------


@pytest.mark.parametrize("missing", ["dataset", "encoder", "features_dir"])
def test_provider_requires_params(tmp_path, missing):
    params = _params(tmp_path)
    params.pop(missing)
    with pytest.raises(RuntimeError, match=missing):
        PrepareFeaturesProvider(config=OmegaConf.create({}), params=params)


def test_provider_rejects_dataset_without_contract(tmp_path):
    params = _params(tmp_path)
    params["dataset"] = {"data_src": FIXTURES, "data_src_args": {}}
    # Point `Dataset` at the contract-less class for this test only.
    original = fixtures.Dataset
    fixtures.Dataset = fixtures.DatasetWithoutContract
    try:
        with pytest.raises(TypeError, match="get_pool_key"):
            PrepareFeaturesProvider(config=OmegaConf.create({}), params=params)
    finally:
        fixtures.Dataset = original


def test_build_env_local_contents(tmp_path):
    provider = PrepareFeaturesProvider(
        config=OmegaConf.create({}), params=_params(tmp_path)
    )
    env = provider.build_env_local()

    assert isinstance(env["dataset"], fixtures.TinyAudioDataset)
    assert isinstance(env["model"], fixtures.DummyEncoder)
    assert env["pool_indices"] == {"spkA": [0, 1, 2], "spkB": [3, 4], "spkC": [5]}
    assert env["features_dir"] == str(tmp_path)
    assert env["prematch"] is True and env["topk"] == 2
    assert isinstance(env["pool_cache"], PoolFeatureCache)
    assert env["pool_cache"].feats == {}


def test_build_worker_setup_fn_matches_local(tmp_path):
    provider = PrepareFeaturesProvider(
        config=OmegaConf.create({}), params=_params(tmp_path)
    )
    env = provider.build_worker_setup_fn()()

    assert env["pool_indices"] == provider.build_env_local()["pool_indices"]
    assert env["prematch"] is True and env["topk"] == 2


def test_build_env_local_reuses_driver_dataset(tmp_path):
    """The driver indexes the corpus once; workers build their own copy."""
    provider = PrepareFeaturesProvider(
        config=OmegaConf.create({}), params=_params(tmp_path)
    )
    env = provider.build_env_local()

    assert env["dataset"] is provider.dataset
    # The worker path must not reuse the driver instance (it is pickled away).
    assert provider.build_worker_setup_fn()()["dataset"] is not provider.dataset
