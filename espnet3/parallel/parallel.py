import inspect
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Generator, Iterable, Optional

import torch
from dask.distributed import (
    Client,
    LocalCluster,
    SSHCluster,
    WorkerPlugin,
    as_completed,
)
from dask_jobqueue import (
    HTCondorCluster,
    LSFCluster,
    MoabCluster,
    OARCluster,
    PBSCluster,
    SGECluster,
    SLURMCluster,
)
from omegaconf import DictConfig
from tqdm import tqdm
from typeguard import typechecked

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


def make_local_gpu_cluster(n_workers: int, options: dict) -> Client:
    """
    Create a Dask LocalCUDACluster using available GPUs.

    This requires `dask_cuda` package.

    Args:
        n_workers (int): Number of Dask workers (must not exceed number of GPUs).
        options (dict): Additional options for the LocalCUDACluster.

    Returns:
        Client: Dask client connected to the GPU cluster.
    """
    try:
        from dask_cuda import LocalCUDACluster
    except:
        raise RuntimeError("Please install dask_cuda.")

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
def set_parallel(config: DictConfig) -> None:
    """Set the global Dask cluster using the provided configuration.

    Args:
        config (DictConfig): Configuration object with 'env' and cluster options.

    Example:
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.create({'env': 'local', 'n_workers': 2})
        >>> set_parallel(config)
    """
    global parallel_config
    options = dict(config.options) if hasattr(config, "options") else {}
    config.options = options
    parallel_config = config


def get_parallel_config() -> Optional[DictConfig]:
    """Return the global Dask cluster configuration."""
    return parallel_config


def _make_client(config: DictConfig = None) -> Client:
    """Create a Dask client tied to the global singleton cluster."""
    if config.env == "local":
        return Client(LocalCluster(config.n_workers, **config.options))

    elif config.env == "local_gpu":
        return make_local_gpu_cluster(config.n_workers, config.options)

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


def make_client(config: DictConfig = None) -> Client:
    """
    Create or retrieve a Dask client using the provided or global configuration.

    Args:
        config (DictConfig, optional): Cluster config. If None, uses global one.

    Returns:
        Client: Dask client instance.
    """
    if config is not None:
        return _make_client(config)

    if parallel_config is None:
        raise ValueError(
            "Parallel configuration not set. Use `set_parallel` to set it."
        )

    return _make_client(parallel_config)


class DictReturnWorkerPlugin(WorkerPlugin):
    """
    A WorkerPlugin that calls a user-defined setup function once per worker,
    and stores the returned dictionary in `worker.plugins["env"]`.
    """

    def __init__(self, setup_fn: Callable[[], dict]):
        self.setup_fn = setup_fn

    def setup(self, worker):
        env = self.setup_fn()
        if not isinstance(env, dict):
            raise ValueError("setup_fn must return a dict")
        worker.plugins["env"] = env


def wrap_func_with_worker_env(func: Callable) -> Callable:
    """
    Wrap a user function to inject environment variables returned from setup_fn.

    This function uses `inspect.signature` to analyze the original function's parameters.
    It will automatically pass only the needed items from `worker.plugins["env"]`,
    while checking for conflicts with explicitly passed arguments.

    Args:
        func: User-defined function. Can take positional and/or keyword args.

    Returns:
        Wrapped function that pulls from the worker's environment if needed.
    """
    sig = inspect.signature(func)
    param_names = set(sig.parameters.keys())

    def wrapped(*args, **kwargs):
        from distributed.worker import get_worker

        env = get_worker().plugins.get("env", {})

        # Detect conflict: both kwargs and env providing the same key
        intersection = param_names & env.keys() & kwargs.keys()
        if intersection:
            raise ValueError(
                f"Argument conflict: {intersection} passed via both kwargs and env"
            )

        # Pick only needed items from env
        filtered_env = {
            k: v for k, v in env.items() if k in param_names and k not in kwargs
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
    client = make_client(config)
    if setup_fn is not None:
        plugin = DictReturnWorkerPlugin(setup_fn)
        client.register_plugin(plugin)
    try:
        yield client
    finally:
        if not isinstance(client, LocalCluster):
            client.shutdown()


@typechecked
def parallel_map(
    func: Callable[[Any], Any], data: Iterable[Any], client: Optional[Client] = None
) -> list:
    """Run a function over a list of data in parallel using Dask.

    Args:
        func (Callable[[Any], Any]): The function to apply.
        data (Iterable[Any]): Iterable of input data.
        client (Optional[Client]): An optional Dask client. If None, uses
            a temporary one.

    Returns:
        list: List of results from applying the function in parallel.

    Example:
        >>> def square(x): return x * x
        >>> results = parallel_map(square, range(5))
    """
    results = []
    if client is not None:
        futures = client.map(func, data)
        return list(tqdm(client.gather(futures), total=len(futures)))
    else:
        with get_client() as client:
            futures = client.map(func, data)
            return list(tqdm(client.gather(futures), total=len(futures)))
    return results


def parallel_for(
    func: Callable,
    args: Iterable,
    client: Optional[Client] = None,
) -> Generator:
    """
    Dispatch parallel tasks with Dask and iterate over results.

    Args:
        func: Function to map over args.
        args: Iterable of inputs to func.
        client: Optional external Dask client. If None, creates one internally.

    Yields:
        Each result (in order if keep_order=True).
    """
    internal = client is None
    if internal:
        client = Client()

    try:
        wrapped_func = wrap_func_with_worker_env(func)
        futures = client.map(wrapped_func, args)
        for future in as_completed(futures):
            yield future.result()
    finally:
        if internal:
            client.close()
