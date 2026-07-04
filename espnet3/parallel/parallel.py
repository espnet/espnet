"""Dask-based parallel processing utilities for ESPnet3."""

import copy
import inspect
import os
import warnings
from contextlib import contextmanager
from typing import Callable, Generator, Optional

import torch
from omegaconf import DictConfig
from typeguard import typechecked

try:
    from dask.distributed import Client, LocalCluster, SSHCluster, WorkerPlugin
    from dask_jobqueue import (
        HTCondorCluster,
        LSFCluster,
        MoabCluster,
        OARCluster,
        PBSCluster,
        SGECluster,
        SLURMCluster,
    )

    _DASK_AVAILABLE = True
except ImportError:
    Client = None
    LocalCluster = None
    SSHCluster = None

    class WorkerPlugin:
        """Lightweight placeholder used when Dask is unavailable."""

        def __init__(self, *args, **kwargs):
            """Create a placeholder WorkerPlugin when Dask is missing."""
            pass

    class _MissingCluster:
        def __init__(self, *args, **kwargs):
            """Raise a helpful error when a Dask cluster type is unavailable."""
            raise RuntimeError(
                "Dask is required for espnet3.parallel; please install dask "
                "and dask_jobqueue to enable parallel features."
            )

    HTCondorCluster = _MissingCluster
    LSFCluster = _MissingCluster
    MoabCluster = _MissingCluster
    OARCluster = _MissingCluster
    PBSCluster = _MissingCluster
    SGECluster = _MissingCluster
    SLURMCluster = _MissingCluster
    _DASK_AVAILABLE = False

try:
    from dask_cuda import LocalCUDACluster
except ImportError:
    LocalCUDACluster = None

parallel_config: Optional[DictConfig] = None

CLUSTER_MAP = {
    "htcondor": HTCondorCluster,
    "lsf": LSFCluster,
    "moab": MoabCluster,
    "oar": OARCluster,
    "pbs": PBSCluster,
    "sge": SGECluster,
    "slurm": SLURMCluster,
    "ssh": SSHCluster,
}


def _ensure_dask():
    if not _DASK_AVAILABLE:
        raise RuntimeError(
            "Dask is not available. Install dask[distributed] and "
            "dask_jobqueue to use espnet3.parallel."
        )


def build_local_gpu_cluster(n_workers: int, options: dict) -> Client:
    """Create a Dask LocalCUDACluster using available GPUs.

    This requires `dask_cuda` package.

    Args:
        n_workers (int): Number of Dask workers (must not exceed number of GPUs).
        options (dict): Additional options for the LocalCUDACluster.

    Returns:
        Client: Dask client connected to the GPU cluster.
    """
    _ensure_dask()
    if LocalCUDACluster is None:
        raise RuntimeError(
            "Please install dask_cuda along with cuda-python and cuda-bindings."
        )

    num_gpus = torch.cuda.device_count()
    if n_workers > num_gpus:
        raise ValueError(f"n_workers={n_workers} > num_gpus={num_gpus}")
    if n_workers < num_gpus:
        warnings.warn(
            f"n_workers={n_workers} < num_gpus={num_gpus}, some GPUs may be idle."
        )

    cluster = LocalCUDACluster(n_workers=n_workers, **options)
    return Client(cluster)


@typechecked
def set_parallel(config: Optional[DictConfig]) -> None:
    """Set the global Dask cluster using the provided configuration.

    Args:
        config (DictConfig): Configuration object with 'env' and cluster options.

    Example:
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.create({'env': 'local', 'n_workers': 2})
        >>> set_parallel(config)
    """
    global parallel_config
    if config is None:
        if parallel_config is not None:
            config = parallel_config
        else:
            config = OmegaConf.create({"env": "local", "n_workers": 1, "options": {}})
    options = dict(config.options) if hasattr(config, "options") else {}
    config.options = options
    parallel_config = copy.copy(config)


def get_parallel_config() -> Optional[DictConfig]:
    """Return the global Dask cluster configuration."""
    return parallel_config


def _build_client(config: DictConfig = None) -> Client:
    """Create a Dask client tied to the global singleton cluster."""
    _ensure_dask()
    if config.env == "local":
        return Client(LocalCluster(n_workers=config.n_workers, **config.options))

    elif config.env == "local_gpu":
        return build_local_gpu_cluster(config.n_workers, config.options)

    elif config.env == "kube":
        try:
            from dask_kubernetes import KubeCluster
        except ImportError:
            raise RuntimeError("Please install dask_kubernetes.")
        cluster = KubeCluster(**config.options)
        cluster.scale(config.n_workers)
        return Client(cluster)

    elif config.env in CLUSTER_MAP:
        cluster = CLUSTER_MAP[config.env](**config.options)
        cluster.scale(config.n_workers)
        return Client(cluster)

    else:
        raise ValueError(f"Unknown env: {config.env}")


def build_client(config: DictConfig = None) -> Client:
    """Create or retrieve a Dask client using the provided or global configuration.

    Args:
        config (DictConfig, optional): Cluster config. If None, uses global one.

    Returns:
        Client: Dask client instance.
    """
    if config is not None:
        set_parallel(config)
        return _build_client(config)

    if parallel_config is None:
        raise ValueError(
            "Parallel configuration not set. Use `set_parallel` to set it."
        )

    return _build_client(parallel_config)


class DictReturnWorkerPlugin(WorkerPlugin):
    """A plugin worker for easily returning a dictionary from a setup function.

    A WorkerPlugin that calls a user-defined setup function once per worker,
    and stores the returned dictionary in `worker.plugins["env"]`.
    """

    def __init__(self, setup_fn: Callable[[], dict]):
        """Initialize the DictReturnWorkerPlugin with a setup function."""
        self.setup_fn = setup_fn

    def setup(self, worker):
        """Set up the worker by calling the user-defined setup function."""
        env = self.setup_fn()
        if not isinstance(env, dict):
            raise ValueError("setup_fn must return a dict")
        worker.plugins["env"] = env

        # Set worker id so that users can use it for identifing workers
        os.environ["DASK_WORKER_ID"] = str(worker.id)


def _register_worker_plugin(client: Client, plugin: WorkerPlugin, name: str) -> None:
    """Register a worker plugin across Dask client API versions."""
    register_worker_plugin = getattr(client, "register_worker_plugin", None)
    if callable(register_worker_plugin):
        register_worker_plugin(plugin, name=name)
        return

    register_plugin = getattr(client, "register_plugin", None)
    if callable(register_plugin):
        register_plugin(plugin, name=name)
        return

    raise RuntimeError(
        "This Dask client lacks worker plugin registration APIs; please upgrade."
    )


def wrap_func_with_worker_env(func: Callable) -> Callable:
    """Wrap a user-defined function for a WorkerPlugin.

    This wrapper inspects the function signature and injects values from the
    worker environment when names match missing keyword parameters.

    Args:
        func (Callable):
            The original user-defined function to be executed on the worker.
            It may have positional parameters, keyword parameters, and/or
            a ``**kwargs`` catch-all.

    Returns:
        Callable: Wrapped callable that pulls matching keyword values from
            ``worker.plugins["env"]`` before invoking ``func``.

    Raises:
        ValueError: Raised when both the worker environment and explicit
            keyword arguments define the same parameter.

    Notes:
        - Only environment keys that match the function's parameter names
          (or any keys if the function accepts ``**kwargs``) will be considered
          for injection.

    Example:
        >>> def process(idx, dataset, model):
        ...     return model(dataset[idx])
        >>> wrapped_fn = wrap_func_with_worker_env(process)
        >>> # On a Dask worker with env={"dataset": ds, "model": md},
        >>> # wrapped_fn(0) injects dataset and model from the worker plugin.
    """
    sig = inspect.signature(func)
    param_names = set(sig.parameters.keys())
    accepts_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    def wrapped(*args, **kwargs):
        from distributed.worker import get_worker

        worker = get_worker()

        env = get_worker().plugins.get("env", {})
        if isinstance(env, DictReturnWorkerPlugin):
            env = env.setup_fn()
            worker.plugins["env"] = env

        kwarg_keys = set(kwargs.keys())
        considered = kwarg_keys if accepts_var_keyword else (param_names & kwarg_keys)
        conflict = set(env.keys()) & considered
        if conflict:
            raise ValueError(
                f"Argument conflict: {conflict} passed via both kwargs and env"
            )

        if accepts_var_keyword:
            filtered_env = {k: v for k, v in env.items() if k not in kwargs}
        else:
            filtered_env = {
                k: v for k, v in env.items() if (k in param_names) and (k not in kwargs)
            }
        return func(*args, **kwargs, **filtered_env)

    return wrapped


@contextmanager
def get_client(
    config: DictConfig = None, setup_fn: Optional[Callable[[], dict]] = None
) -> Generator[Client, None, None]:
    """Context manager to yield a Dask client from the global singleton cluster.

    Args:
        config (DictConfig, optional): Cluster config.
        setup_fn (Callable[[], dict], optional): A setup function that runs
            on each worker and returns a dictionary of environment variables.

    Yields:
        Client: A Dask client instance tied to the global cluster.

    Example:
        >>> with get_client() as client:
        ...     results = client.map(lambda x: x**2, range(10))
    """
    client = build_client(config)
    if setup_fn is not None:
        plugin = DictReturnWorkerPlugin(setup_fn)
        _register_worker_plugin(client, plugin, name="env")
    try:
        yield client
    finally:
        # Avoid shutdown for LocalCluster and LocalCUDACluster
        cluster = getattr(client, "cluster", None)

        # always close the client first
        client.close()

        if cluster is not None:
            close = getattr(cluster, "close", None)
            if callable(close):
                close()
            else:
                shutdown = getattr(cluster, "shutdown", None)
                if callable(shutdown):
                    shutdown()
