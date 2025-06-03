import os
from contextlib import contextmanager
from typing import Any, Callable, Generator, Iterable, Optional

from dask.distributed import Client, LocalCluster, WorkerPlugin
from dask_jobqueue import SLURMCluster
from omegaconf import DictConfig
from tqdm import tqdm
from typeguard import typechecked

import torch
import torch.multiprocessing as mp
import warnings


parallel_config: Optional[DictConfig] = None


class CUDADevicePlugin(WorkerPlugin):
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id

    def setup(self, worker):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)


def make_local_gpu_cluster(n_workers: int, options: dict) -> Client:
    num_gpus = torch.cuda.device_count()
    if n_workers > num_gpus:
        raise ValueError(f"n_workers={n_workers} > num_gpus={num_gpus}")
    if n_workers < num_gpus:
        import warnings
        warnings.warn(f"n_workers={n_workers} < num_gpus={num_gpus}, some GPUs may be idle.")

    cluster = LocalCluster(n_workers=n_workers, **options)
    client = Client(cluster)

    for i, worker in enumerate(client.scheduler_info()["workers"].values()):
        plugin = CUDADevicePlugin(gpu_id=i)
        client.register_worker_plugin(plugin, name=f"cuda-device-{i}")

    return client


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
        return LocalCluster(config.n_workers, **config.options)

    if config.env == "local-gpu":
        return make_local_gpu_cluster(config.n_workers, config.options)
    
    elif config.env == "slurm":
        cluster = SLURMCluster(**config.options)
        cluster.scale(config.n_workers)
        return Client(cluster)

    else:
        raise ValueError(f"Unknown env: {config.env}")


def make_client(config: DictConfig = None) -> Client:
    """Create a Dask client tied to the global singleton cluster."""
    if config is not None:
        return _make_client(config)

    if parallel_config is None:
        raise ValueError(
            "Parallel configuration not set. Use `set_parallel` to set it."
        )

    return _make_client(parallel_config)


@contextmanager
def get_client(
    config: DictConfig = None, plugin: WorkerPlugin = None
) -> Generator[Client, None, None]:
    """Context manager to yield a Dask client from the global singleton cluster.

    Yields:
        Client: A Dask client instance tied to the global cluster.

    Example:
        >>> with get_client() as client:
        ...     results = client.map(lambda x: x**2, range(10))
    """
    client = make_client(config)
    if plugin is not None:
        client.register_worker_plugin(plugin)
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


@typechecked
def parallel_submit(
    func: Callable[..., Any], *args: Any, client: Optional[Client] = None, **kwargs: Any
) -> Any:
    """Submit a single function call asynchronously using Dask.

    Args:
        func (Callable[..., Any]): The function to run.
        *args (Any): Positional arguments to the function.
        client (Optional[Client]): Optional Dask client.
        **kwargs (Any): Keyword arguments to the function.

    Returns:
        Future: A future representing the asynchronous computation.

    Example:
        >>> future = parallel_submit(pow, 2, 3)
        >>> future.result()  # returns 8
    """
    if client is not None:
        return client.submit(func, *args, **kwargs)
    else:
        with get_client() as client:
            return client.submit(func, *args, **kwargs).result()


@typechecked
def parallel_scatter(data: Any, client: Optional[Client] = None) -> Any:
    """Scatter data to workers for shared access in distributed computing.

    Args:
        data (Any): Data to scatter.
        client (Optional[Client]): Optional Dask client.

    Returns:
        Future: A future or list of futures for scattered data.

    Example:
        >>> scattered_data = parallel_scatter([1, 2, 3])
    """
    if client is not None:
        return client.scatter(data)
    else:
        with get_client() as client:
            return client.scatter(data)
