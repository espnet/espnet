import multiprocessing as mp

import pytest
from omegaconf import OmegaConf

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
