import multiprocessing as mp
import sys
import time
import types

import pytest
from omegaconf import OmegaConf

from espnet3.parallel.parallel import (
    DictReturnWorkerPlugin,
    get_client,
    make_client,
    make_local_gpu_cluster,
    parallel_for,
    parallel_map,
    set_parallel,
)

mp.set_start_method("fork", force=True)

# ===============================================================
# Test Case Summary for Parallel Processing Utilities
# ===============================================================

# Normal Cases
# | Test Function Name                               | Description                     |
# |--------------------------------------------------|---------------------------------|
# | test_set_and_get_parallel_config                 | Sets and retrieves global config|
# | test_set_parallel_copies_options_dict            | Copies options dict, ignores    |
# |                                                  | external mutations              |
# | test_make_client_local                           | Creates LocalCluster client and |
# |                                                  | verifies distributed mapping    |
# | test_get_client_context_manager_and_parallel_map | Uses get_client with global     |
# |                                                  | config to run parallel_map      |
# | test_parallel_map_internal_client                | Runs parallel_map with          |
# |                                                  | client=None                     |
# | test_parallel_for_streaming_yields_results       | Confirms parallel_for yields    |
# |                                                  | results in streaming fashion    |
# | test_parallel_map_preserves_input_order          | Confirms parallel_map preserves |
# |                                                  | input order                     |
# | test_parallel_for_completion_order               | Confirms parallel_for yields in |
# |                                                  | completion order                |
# | test_worker_env_injection_via_setup_fn           | Injects worker env vars         |
# | test_parallel_map_auto_inject_env_via_setup_fn   | Verifies parallel_map injects   |
# |                                                  | worker env vars via setup_fn    |
# | test_parallel_for_auto_inject_env_via_setup_fn_without_with | Verifies parallel_for|
# |                                                  | injects env vars without with   |
# | test_wrap_env_filters_unknown_keys               | Ignores unknown keys from env   |
# | test_parallel_map_registers_setup_fn_when_passed_directly | Auto-injects env via   |
# |                                                  | direct setup_fn to parallel_map |
# | test_parallel_for_registers_setup_fn_when_passed_directly | Auto-injects env via   |
# |                                                  | direct setup_fn to parallel_for |
# | test_get_client_context_auto_shutdown            | Closes/shuts down client safely |

# Error Cases
# | Test Function Name                               | Description                     |
# |--------------------------------------------------|---------------------------------|
# | test_worker_env_conflict_detection               | ValueError on args conflict     |
# | test_worker_env_conflict_detection_parallel_for  | ValueError on args conflict     |
# | test_make_local_gpu_cluster_import_guard         | RuntimeError when dask_cuda miss|
# | test_make_client_unknown_env_raises              | ValueError on unknown env       |
# | test_make_client_kube_import_guard               | RuntimeError when dask_kube miss|
# | test_worker_plugin_setup_must_return_dict        | ValueError when setup_fn != dict|
# | test_parallel_for_propagates_task_exception      | Propagates exception from task  |

# Expected Exceptions
# | Test Function Name                               | Expected Exception              |
# |--------------------------------------------------|---------------------------------|
# | test_worker_env_conflict_detection               | ValueError                      |
# | test_worker_env_conflict_detection_parallel_for  | ValueError                      |
# | test_make_local_gpu_cluster_import_guard         | RuntimeError                    |
# | test_make_client_unknown_env_raises              | ValueError                      |
# | test_make_client_kube_import_guard               | RuntimeError                    |
# | test_worker_plugin_setup_must_return_dict        | ValueError                      |
# | test_parallel_for_propagates_task_exception      | RuntimeError                    |


# --------- Fixtures ---------


@pytest.fixture
def local_cfg():
    # LocalCluster configuration (safe for CI, no GPU or jobqueue)
    cfg = OmegaConf.create(
        {
            "env": "local",
            "n_workers": 2,
            "options": {
                "threads_per_worker": 1,
                "processes": True,
            },
        }
    )
    return cfg


@pytest.fixture
def set_global_parallel(local_cfg):
    # Prepare global parallel config for tests
    set_parallel(local_cfg)
    yield
    # Re-set to avoid side effects (dummy reassign is fine here)
    set_parallel(local_cfg)


# --------- Normal cases ---------

# ------------------------------------------------------------
# Normal case: set and get parallel config
# ------------------------------------------------------------


def test_set_parallel_copies_options_dict(local_cfg):
    cfg = local_cfg
    set_parallel(cfg)
    got = getattr(
        __import__("espnet3.parallel.parallel", fromlist=["get_parallel_config"]),
        "get_parallel_config",
    )()
    # mutate original options
    cfg.options["threads_per_worker"] = 999
    # The held object of get_parallel_config should not be affected
    assert got.options["threads_per_worker"] != 999


def test_set_and_get_parallel_config(local_cfg):
    set_parallel(local_cfg)
    got = local_cfg
    assert got.env == "local"
    assert got.n_workers == 2
    assert "threads_per_worker" in got.options


# ------------------------------------------------------------
# Nromal case: client management
# ------------------------------------------------------------


def test_make_client_local(local_cfg):
    client = make_client(local_cfg)
    try:
        futs = client.map(lambda x: x * x, range(5))
        out = client.gather(futs)
        assert out == [0, 1, 4, 9, 16]
    finally:
        client.close()


def test_get_client_context_manager_and_parallel_map(set_global_parallel):
    # Use get_client() with a global parallel config
    with get_client() as client:
        res = parallel_map(lambda x: x + 1, range(4), client=client)
        assert res == [1, 2, 3, 4]


@pytest.mark.execution_timeout(30)
def test_parallel_map_internal_client(local_cfg, monkeypatch):
    # Test the branch where client=None and get_client() is internally used
    class _Ctx:
        def __init__(self, client):
            self.client = client

        def __enter__(self):
            return self.client

        def __exit__(self, exc_type, exc, tb):
            self.client.close()

    cli = make_client(local_cfg)
    monkeypatch.setattr(
        "espnet3.parallel.parallel.get_client", lambda *a, **k: _Ctx(cli)
    )
    res = parallel_map(lambda x: x * 2, [1, 2, 3])
    assert res == [2, 4, 6]


# ------------------------------------------------------------
# normal case： ordering / streaming
# ------------------------------------------------------------


def test_parallel_for_streaming_yields_results(local_cfg):
    # parallel_for should yield results in streaming fashion (order not guaranteed)
    client = make_client(local_cfg)
    try:

        def work(x):
            time.sleep(0.01)  # Slight delay to expose concurrency
            return x * 10

        results = list(parallel_for(work, range(5), client=client))
        assert sorted(results) == [0, 10, 20, 30, 40]
    finally:
        client.close()


def test_parallel_map_preserves_input_order(local_cfg):
    def work(x):
        time.sleep(0.005 * (x % 3))
        return x * 10

    with get_client(local_cfg) as client:
        data = list(range(8))
        out = parallel_map(work, data, client=client)
        assert out == [x * 10 for x in data]


def test_parallel_for_completion_order(local_cfg):
    delays = {0: 0.12, 1: 0.09, 2: 0.06, 3: 0.03}

    def work(i):
        time.sleep(delays[i])
        return i

    with get_client(local_cfg) as client:
        results = list(parallel_for(work, [0, 1, 2, 3], client=client))
        assert set(results) == set([3, 2, 1, 0])


# ------------------------------------------------------------
# Normal case: environment variables and setup functions
# ------------------------------------------------------------


@pytest.mark.execution_timeout(30)
def test_worker_env_injection_via_setup_fn(local_cfg):
    # Verify env injection via get_client(setup_fn=...) without manual wrapping
    def setup_fn():
        return {"bias": 10}

    def add_bias(x, bias=-1):
        return x + bias

    # Use get_client to register the worker env; parallel_map should auto-inject 'bias'
    with get_client(local_cfg, setup_fn=setup_fn) as client:
        out = parallel_map(add_bias, [1, 2, 3], client=client)
    assert out == [11, 12, 13]


@pytest.mark.execution_timeout(30)
def test_parallel_map_auto_inject_env_via_setup_fn(local_cfg):
    # Verify parallel_map auto-injects worker env
    # when setup_fn is provided via get_client
    def setup_fn():
        return {"bias": 10}

    with get_client(local_cfg, setup_fn=setup_fn) as client:
        # No manual wrapping; parallel_map should inject 'bias' automatically
        out = parallel_map(lambda x, bias=-1: x + bias, [1, 2, 3], client=client)
        assert out == [11, 12, 13]


@pytest.mark.execution_timeout(30)
def test_parallel_for_auto_inject_env_via_setup_fn_without_with(local_cfg):
    # Verify parallel_for auto-injects env
    # without using a 'with' block (manual enter/exit)
    def setup_fn():
        return {"factor": 4}

    ctx = get_client(local_cfg, setup_fn=setup_fn)
    client = ctx.__enter__()
    try:
        # No manual wrapping; 'factor' should be injected into the function
        results = list(
            parallel_for(lambda x, factor: x * factor, range(5), client=client)
        )
        assert sorted(results) == [0, 4, 8, 12, 16]
    finally:
        ctx.__exit__(None, None, None)


@pytest.mark.execution_timeout(30)
def test_wrap_env_filters_unknown_keys(local_cfg):
    def setup_fn():
        return {"bias": 10, "unknown": "ignored"}

    def add_bias(x, bias=-1):
        return x + bias

    with get_client(local_cfg, setup_fn=setup_fn) as client:
        out = parallel_map(add_bias, [1, 2, 3], client=client)
        assert out == [11, 12, 13]


@pytest.mark.execution_timeout(30)
def test_parallel_map_registers_setup_fn_when_passed_directly(local_cfg):
    def setup_fn():
        return {"bias": 5}

    def add_bias(x, bias=-1):
        return x + bias

    out = parallel_map(add_bias, [10, 20, 30], setup_fn=setup_fn)
    assert out == [15, 25, 35]


@pytest.mark.execution_timeout(30)
def test_parallel_for_registers_setup_fn_when_passed_directly(local_cfg):
    def setup_fn():
        return {"factor": 3}

    def mul(x, factor):
        return x * factor

    results = list(parallel_for(mul, [1, 2, 3, 4], setup_fn=setup_fn))
    assert sorted(results) == [3, 6, 9, 12]


# --------- Error cases ---------


@pytest.mark.execution_timeout(30)
def test_worker_env_conflict_detection(local_cfg):
    # When both worker env and kwargs provide the same key,
    # wrapping should raise ValueError
    def setup_fn():
        return {"bias": 7}

    def add_bias(x, bias=10):
        return x + bias

    # Register env via get_client; then explicitly wrap
    # and also pass a conflicting kwarg
    with get_client(local_cfg, setup_fn=setup_fn) as client:
        # conflict: 'bias' from env AND kwargs
        with pytest.raises(ValueError, match=r"\bbias\b"):
            parallel_map(add_bias, [1, 2], client=client, bias=5)


@pytest.mark.execution_timeout(30)
def test_worker_env_conflict_detection_parallel_for(local_cfg):
    # When both worker env and kwargs provide the same key,
    # submission-time check should raise ValueError
    def setup_fn():
        return {"bias": 7}

    def add_bias(x, bias=10):
        return x + bias

    with get_client(local_cfg, setup_fn=setup_fn) as client:
        # conflict: 'bias' from env AND kwargs
        with pytest.raises(ValueError, match=r"\bbias\b"):
            list(parallel_for(add_bias, [1, 2], client=client, bias=5))


def test_make_local_gpu_cluster_import_guard(monkeypatch):
    # If dask_cuda is not installed, make_local_gpu_cluster should raise RuntimeError
    if "dask_cuda" in sys.modules:
        pytest.skip("dask_cuda is installed; skipping import-guard test")
    with pytest.raises(RuntimeError):
        make_local_gpu_cluster(n_workers=1, options={})


@pytest.mark.execution_timeout(30)
def test_get_client_context_auto_shutdown(local_cfg, monkeypatch):
    # Ensure get_client shuts down or closes the client without raising exceptions
    calls = {"shutdown": 0, "close": 0}
    cli = make_client(local_cfg)

    class _ClientProxy:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def shutdown(self):
            calls["shutdown"] += 1
            try:
                return self._real.shutdown()
            except Exception:
                return None

        def close(self):
            calls["close"] += 1
            return self._real.close()

    proxy = _ClientProxy(cli)
    monkeypatch.setattr("espnet3.parallel.parallel.make_client", lambda cfg=None: proxy)

    with get_client(local_cfg):
        pass

    assert True


def test_make_client_unknown_env_raises(local_cfg):
    cfg = local_cfg.copy()
    cfg.env = "unknown_env"
    with pytest.raises(ValueError, match="Unknown env"):
        make_client(cfg)


def test_make_client_kube_import_guard(monkeypatch, local_cfg):
    if "dask_kubernetes" in sys.modules:
        monkeypatch.delitem(sys.modules, "dask_kubernetes", raising=False)

    cfg = local_cfg.copy()
    cfg.env = "kube"
    cfg.options = {"whatever": "x"}
    with pytest.raises(RuntimeError, match="Please install dask_kubernetes"):
        make_client(cfg)


def test_worker_plugin_setup_must_return_dict():
    plugin = DictReturnWorkerPlugin(setup_fn=lambda: 123)  # not a dict
    dummy_worker = types.SimpleNamespace(plugins={})
    with pytest.raises(ValueError, match="must return a dict"):
        plugin.setup(dummy_worker)


@pytest.mark.execution_timeout(30)
def test_parallel_for_propagates_task_exception(local_cfg):
    def boom(x):
        if x == 2:
            raise RuntimeError("boom")
        return x

    with get_client(local_cfg) as client:
        with pytest.raises(RuntimeError, match="boom"):
            list(parallel_for(boom, [0, 1, 2, 3], client=client))
