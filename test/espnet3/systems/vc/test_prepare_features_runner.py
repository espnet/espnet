"""Tests for the prepare_features runner."""

import threading
from pathlib import Path
from test.espnet3.systems.vc import fixtures

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from espnet3.parallel.parallel import set_parallel
from espnet3.systems.vc.models.knnvc.matcher import match_features
from espnet3.systems.vc.prepare_features_provider import PrepareFeaturesProvider
from espnet3.systems.vc.prepare_features_runner import (
    FEATURE_SUFFIX,
    PoolFeatureCache,
    PrepareFeaturesRunner,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_forward_without_prematch_encodes_padded| Frame count == padded        |
# |                                             | samples / hop.               |
# | test_forward_prematch_matches_reference     | Prematched feats equal a     |
# |                          | direct kNN against the other same-pool utts.    |
# | test_forward_single_speaker_falls_back      | Lone-pool utterance keeps    |
# |                                             | raw features.                |
# | test_runner_call_writes_features_and_scp    | End-to-end: npy per utt,     |
# |                          | merged feats.<name>.scp, resume skips done shards|
# | test_pool_cache_is_bounded_to_one_pool      | Cache holds only the current |
# |                                             | pool's features.             |
# | test_pool_cache_is_thread_local             | Concurrent threads never see |
# |                          | each other's pool (would corrupt matching pools).|


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
# PrepareFeaturesRunner
# ---------------------------------------------------------------


def _env(tmp_path, **overrides):
    provider = PrepareFeaturesProvider(
        config=OmegaConf.create({}), params=_params(tmp_path, **overrides)
    )
    return provider.build_env_local()


def test_forward_without_prematch_encodes_padded(tmp_path):
    env = _env(tmp_path, prematch=False)
    result = PrepareFeaturesRunner.forward(0, **env)

    samples = len(env["dataset"][0]["speech"])
    # Like the official prematch script, a full hop is padded even when the
    # waveform length is already a multiple of the hop.
    expected_frames = samples // fixtures.HOP_LENGTH + 1
    assert result["feature_name"] == "train/spkA/spkA-0001"
    assert result["pool_key"] == "spkA"
    assert result["feats"].dtype == np.float16
    assert result["feats"].shape == (expected_frames, fixtures.FEATURE_DIM)


def test_forward_prematch_matches_reference(tmp_path):
    env = _env(tmp_path)
    result = PrepareFeaturesRunner.forward(1, **env)  # spkA-0002

    encoder = env["model"]
    dataset = env["dataset"]
    source = encoder.encode(dataset[1]["speech"], pad_to_hop=True).half().float()
    pool = torch.cat(
        [
            encoder.encode(dataset[i]["speech"], pad_to_hop=True).half().float()
            for i in (0, 2)
        ]
    )
    expected = match_features(source, pool, topk=2).half().numpy()
    np.testing.assert_allclose(result["feats"], expected, rtol=1e-3, atol=1e-3)


def test_forward_single_speaker_falls_back(tmp_path, caplog):
    env = _env(tmp_path)
    with caplog.at_level("WARNING"):
        result = PrepareFeaturesRunner.forward(5, **env)  # spkC, one utterance

    raw = env["model"].encode(env["dataset"][5]["speech"], pad_to_hop=True)
    np.testing.assert_allclose(
        result["feats"], raw.half().numpy(), rtol=1e-3, atol=1e-3
    )
    assert "single utterance" in caplog.text


def test_pool_cache_is_bounded_to_one_pool(tmp_path):
    env = _env(tmp_path)
    PrepareFeaturesRunner.forward(0, **env)
    assert env["pool_cache"].pool_key == "spkA"
    assert set(env["pool_cache"].feats) == {0, 1, 2}

    PrepareFeaturesRunner.forward(3, **env)
    assert env["pool_cache"].pool_key == "spkB"
    assert set(env["pool_cache"].feats) == {3, 4}


def test_pool_cache_is_thread_local(tmp_path):
    """Two threads on one worker must not share a pool (see PoolFeatureCache).

    A Dask worker runs several task threads, so two shards covering different
    pools can call ``forward`` concurrently with the same env dict. If the
    cache were shared, one pool's features could land in another pool's
    matching set and the written features would be silently wrong.
    """
    env = _env(tmp_path)
    assert isinstance(env["pool_cache"], PoolFeatureCache)
    seen = {}
    barrier = threading.Barrier(2)

    def run(name, idx):
        barrier.wait()
        result = PrepareFeaturesRunner.forward(idx, **env)
        seen[name] = (
            env["pool_cache"].pool_key,
            set(env["pool_cache"].feats),
            result["pool_key"],
        )

    threads = [
        threading.Thread(target=run, args=("a", 0)),
        threading.Thread(target=run, args=("b", 3)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen["a"] == ("spkA", {0, 1, 2}, "spkA")
    assert seen["b"] == ("spkB", {3, 4}, "spkB")


def test_runner_call_writes_features_and_scp(tmp_path):
    features_dir = tmp_path / "features"
    provider = PrepareFeaturesProvider(
        config=OmegaConf.create({}), params=_params(features_dir)
    )
    runner = PrepareFeaturesRunner(
        provider=provider,
        output_dir=features_dir / "shards",
        shard_subdir="train",
    )
    scp_path = runner(range(len(provider.dataset)))

    assert scp_path == features_dir / "feats.train.scp"
    lines = scp_path.read_text().strip().splitlines()
    assert len(lines) == len(fixtures.UTTERANCES)
    for line in lines:
        name, path = line.split(maxsplit=1)
        assert Path(path) == features_dir / (name + FEATURE_SUFFIX)
        assert Path(path).is_file()
        assert np.load(path).shape[1] == fixtures.FEATURE_DIM
    assert not list(features_dir.rglob("*.tmp.npy"))

    # Second call resumes: done shard is skipped, merged SCP is rebuilt.
    scp_path_again = runner(range(len(provider.dataset)))
    assert scp_path_again.read_text() == scp_path.read_text()
