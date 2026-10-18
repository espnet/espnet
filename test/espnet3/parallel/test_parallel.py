import multiprocessing as mp
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.parallel import (
    _DASK_AVAILABLE,
    DictReturnWorkerPlugin,
    build_client,
    get_client,
    get_parallel_config,
    set_parallel,
    wrap_func_with_worker_env,
)

mp.set_start_method("fork", force=True)

pytestmark = pytest.mark.skipif(not _DASK_AVAILABLE, reason="Dask is not installed")


def _square(x):
    return x * x


def _add_bias(x, bias):
    return x + bias


def _worker_env():
    return {"bias": 4}


def _empty_worker_env():
    return {}


class LocalRunnerProvider:
    """Minimal provider used to exercise BaseRunner's local Dask path."""

    def build_env_local(self):
        return {}

    def build_worker_setup_fn(self):
        return _empty_worker_env


class LocalRunner(BaseRunner):
    """Persist input indices so the test can validate local shard dispatch."""

    @staticmethod
    def forward(idx, **_kwargs):
        return idx

    @staticmethod
    def open_writers(shard_dir, **_kwargs):
        return {"path": Path(shard_dir) / "records.txt", "records": []}

    @staticmethod
    def write_record(writers, result, state, **_kwargs):
        writers["records"].append(str(result))

    @staticmethod
    def close_writers(writers, state, **_kwargs):
        writers["path"].write_text(
            "\n".join(writers["records"]) + "\n", encoding="utf-8"
        )

    def merge(self, shard_dirs):
        return [
            int(line)
            for shard_dir in shard_dirs
            for line in (Path(shard_dir) / "records.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        ]


@pytest.fixture
def local_cfg():
    return OmegaConf.create(
        {
            "env": "local",
            "n_workers": 2,
            "options": {
                "threads_per_worker": 1,
                "processes": True,
            },
        }
    )


def test_set_parallel_copies_options_dict(local_cfg):
    set_parallel(local_cfg)
    got = get_parallel_config()
    local_cfg.options["threads_per_worker"] = 999
    assert got.options["threads_per_worker"] != 999


def test_build_client_local(local_cfg):
    client = build_client(local_cfg)
    try:
        futs = client.map(_square, range(5))
        out = client.gather(futs)
        assert out == [0, 1, 4, 9, 16]
    finally:
        client.close()


def test_base_runner_uses_local_cluster_workers(local_cfg, tmp_path):
    """``env: local`` must use ``n_workers`` instead of falling back to serial."""
    set_parallel(local_cfg)
    output_dir = tmp_path / "runner"

    result = LocalRunner(LocalRunnerProvider(), output_dir=output_dir, resume=False)(
        range(4)
    )

    assert sorted(result) == [0, 1, 2, 3]
    assert sorted(path.name for path in output_dir.glob("split.*")) == [
        "split.0",
        "split.1",
    ]


def test_get_client_registers_worker_env(local_cfg):
    with get_client(local_cfg, setup_fn=_worker_env) as client:
        futs = client.map(
            wrap_func_with_worker_env(_add_bias),
            [1, 2, 3],
        )
        assert client.gather(futs) == [5, 6, 7]


def test_worker_env_conflict_detection(local_cfg):
    with get_client(local_cfg, setup_fn=_worker_env) as client:
        futs = client.map(
            wrap_func_with_worker_env(_add_bias),
            [1],
            bias=10,
        )
        with pytest.raises(ValueError, match="Argument conflict"):
            client.gather(futs)


def test_worker_plugin_setup_must_return_dict():
    plugin = DictReturnWorkerPlugin(setup_fn=lambda: 123)

    class DummyWorker:
        plugins = {}
        id = "dummy"

    with pytest.raises(ValueError, match="setup_fn must return a dict"):
        plugin.setup(DummyWorker())
