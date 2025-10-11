# base_runner.py
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence
from uuid import uuid4

from dask.distributed import Client, wait
from tqdm import tqdm

from espnet3.parallel.parallel import (
    get_client,
    get_parallel_config,
    parallel_map,
    wrap_func_with_worker_env,
)
from espnet3.runner.env_provider import EnvironmentProvider


@dataclass(frozen=True)
class AsyncJobSpec:
    """Specification of an asynchronous shard submitted to the Dask cluster.

    Attributes:
        runner_cls (str): Import path to the concrete ``BaseRunner`` subclass.
        provider_cls (str): Import path to the concrete ``EnvironmentProvider`` subclass
        config (Dict): A resolved, plain-``dict`` serialization of the Hydra config.
        params (Dict): Extra parameters forwarded to the provider/environment.
        indices (List[int]): The subset of indices this shard should process.
        world_size (int): Total number of shards.
        world_rank (int): The shard rank (0..world_size-1).
        result_path (str | None): If set, results are written to this JSONL file
            *on the worker* and not returned to the driver.
        extras (Dict | None): Reserved for future extensions.

    Notes:
        - Each shard is independent and reconstructs both provider and runner
          by import path on the worker before executing :py:meth:`BaseRunner.forward`.
    """

    runner_cls: str  # e.g., module.path.to.your.RunnerClass
    provider_cls: str  # e.g., module.path.to.your.ProviderClass
    config: Dict
    params: Dict
    indices: List[int]
    world_size: int
    world_rank: int
    result_path: str | None
    extras: Dict | None


def _import_obj(class_path: str):
    mod, _, name = class_path.rpartition(".")
    return getattr(importlib.import_module(mod), name)


def _default_chunk(indices: Sequence[int], num_chunks: int) -> List[List[int]]:
    n = max(1, num_chunks)
    L = len(indices)
    q, r = divmod(L, n)
    out = []
    start = 0
    for i in range(n):
        size = q + (1 if i < r else 0)
        out.append(list(indices[start : start + size]))
        start += size
    return [c for c in out if c]  # 空チャンク除去


def convert_paths(obj):
    if isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


class BaseRunner:
    """A thin orchestration layer to run static ``forward`` over indices.

    This class handles:
        - Switching among local, parallel, and asynchronous (distributed) modes.
        - Injecting per-worker environments supplied by an :class:`EnvironmentProvider`.
        - Keeping ``forward`` as a ``@staticmethod`` for pickle-safety.

    Subclass contract:
        - Implement ``@staticmethod forward(idx, *, dataset, model, **env) -> Any``
          without capturing ``self``.
        - Provide an :class:`EnvironmentProvider` that builds the required env
          (e.g., dataset/model) for local and worker executions.

    Args:
        env_provider (EnvironmentProvider): Provider that builds the runtime env.
        async_mode (bool): If True, use Dask ``submit`` with asynchronous shards.
        async_specs_dir (str | Path): Output directory for per-shard spec JSON files.
        async_return_results (bool): If False, results are written to JSONL files
            on workers and the driver returns ``None``.
        async_num_workers (int | None): If set, overrides detected worker count
            to decide how many shards to create.
        async_result_dir (str | Path): Output directory for per-shard JSONL results
            when ``async_return_results=False``.

    Notes:
        - In parallel sync mode (when a Dask cluster is configured), tasks are
          submitted via ``parallel_map`` and results are gathered in order.
        - In async mode, the driver can return immediately if
          ``async_return_results=False`` and results are written on workers.
    """

    def __init__(
        self,
        env_provider: EnvironmentProvider,
        *,
        async_mode: bool = False,
        async_specs_dir: str | Path = "./_async_specs",
        async_return_results: bool = True,
        async_num_workers: int | None = None,
        async_result_dir: str | Path = "./_async_results",
    ):
        self.env_provider = env_provider
        self.async_mode = async_mode
        self.async_specs_dir = Path(async_specs_dir)
        self.async_return_results = async_return_results
        self.async_num_workers = async_num_workers
        self.async_result_dir = Path(async_result_dir)

    @staticmethod
    def forward(idx: int, *, dataset, model, **env) -> Any:
        """Compute one item for the given index (to be implemented by subclasses).

        Keep this as a ``@staticmethod`` so that it is pickle-safe for Dask
        and does not capture ``self``.

        Args:
            idx (int): The input index to process.
            dataset: Dataset object provided via the environment.
            model: Model object provided via the environment.
            **env: Any additional environment entries injected by the provider.

        Returns:
            Any: Result for the given index.

        Raises:
            NotImplementedError: Always in the base class; implement in subclass.

        Example:
            >>> class MyRunner(BaseRunner):
            ...     @staticmethod
            ...     def forward(idx, *, dataset, model, **env):
            ...         x = dataset[idx]
            ...         return model(x)
        """
        raise NotImplementedError

    def _run_local(self, indices: Sequence[int]) -> List[Any]:
        """Run sequentially on the driver using a locally built environment.

        Args:
            indices (Sequence[int]): Indices to process.

        Returns:
            List[Any]: Results in the same order as ``indices``.

        Notes:
            - Uses ``tqdm`` progress bar over the input sequence.
        """
        env = self.env_provider.build_env_local()
        f = self.__class__.forward  # static
        return [f(i, **env) for i in tqdm(indices, total=len(indices))]

    def _run_parallel(self, indices: Sequence[int]) -> List[Any]:
        """Run with synchronous Dask mapping using per-worker environments.

        Args:
            indices (Sequence[int]): Indices to process.

        Returns:
            List[Any]: Results gathered in order.

        Notes:
            - Wraps ``forward`` with :func:`wrap_func_with_worker_env` so that
              missing keyword args are injected from the worker env.
        """
        setup_fn = self.env_provider.make_worker_setup_fn()
        f = wrap_func_with_worker_env(self.__class__.forward)
        return parallel_map(f, indices, setup_fn=setup_fn)

    def _run_async(self, indices: Sequence[int]) -> List[Any] | None:
        """Submit shards to a Dask cluster and gather or write results.

        Workflow:
            1) Emit per-shard JSON specs to ``async_specs_dir``.
            2) Submit worker entrypoints that reconstruct runner/provider and run.
            3) If ``async_return_results`` is True, wait & gather. Otherwise return
               ``None`` immediately and let workers write JSONL files.

        Args:
            indices (Sequence[int]): Indices to process.

        Returns:
            List[Any] | None: Flattened results if gathered; otherwise ``None``.

        Raises:
            RuntimeError: If Dask client creation fails.
            ValueError: If async sharding parameters are invalid.

        Notes:
            - When not gathering, each shard writes to
              ``async_result_dir / f"result-<job_id>.jsonl"`` on its worker.
        """
        par_cfg = get_parallel_config()
        with get_client(par_cfg) as client:
            n_workers = self.async_num_workers or max(
                1, len(client.scheduler_info().get("workers", {})) or 1
            )
            chunks = _default_chunk(indices, n_workers)

            self.async_specs_dir.mkdir(parents=True, exist_ok=True)
            self.async_result_dir.mkdir(parents=True, exist_ok=True)

            # DictConfig -> dict
            from omegaconf import OmegaConf

            cfg_dict = OmegaConf.to_container(self.env_provider.config, resolve=True)

            provider_cls = (
                f"{self.env_provider.__class__.__module__}"
                f".{self.env_provider.__class__.__name__}"
            )
            runner_cls = f"{self.__class__.__module__}.{self.__class__.__name__}"

            futures = []
            for rank, chunk in enumerate(chunks):
                job_id = f"{uuid4().hex[:8]}-r{rank}"
                result_path = (
                    (self.async_result_dir / f"result-{job_id}.jsonl").as_posix()
                    if not self.async_return_results
                    else None
                )

                spec = AsyncJobSpec(
                    runner_cls=runner_cls,
                    provider_cls=provider_cls,
                    config=cfg_dict,
                    params=getattr(self.env_provider, "params", {}) or {},
                    indices=list(chunk),
                    world_size=len(chunks),
                    world_rank=rank,
                    result_path=str(result_path),
                    extras={},
                )
                spec_path = self.async_specs_dir / f"spec-{job_id}.json"
                with open(spec_path, "w", encoding="utf-8") as f:
                    json.dump(convert_paths(asdict(spec)), f, ensure_ascii=False)

                fut = client.submit(
                    _async_worker_entry_from_spec_path,
                    spec_path.as_posix(),
                )
                futures.append(fut)

        if not self.async_return_results:
            return None

        # gather
        wait(futures)
        results = client.gather(futures)
        flat: List[Any] = []
        for r in results:
            if r is None:
                continue
            flat.extend(r)
        return flat

    def __call__(self, indices: Iterable[int]) -> List[Any] | None:
        """Dispatch execution according to the configured parallel mode.

        Args:
            indices (Iterable[int]): Indices to process.

        Returns:
            List[Any] | None: Results for local/parallel modes, or ``None`` in
            async mode when ``async_return_results=False``.

        Notes:
            - If no parallel config is set or ``env='local'``, run locally.
            - Otherwise, run in parallel with environment injection.
        """
        indices = list(indices)
        if self.async_mode:
            return self._run_async(indices)

        par_cfg = get_parallel_config()
        if par_cfg is None or getattr(par_cfg, "env", "local") == "local":
            return self._run_local(indices)
        return self._run_parallel(indices)


def _async_worker_entry_from_spec_path(spec_path: str):
    """Worker entrypoint: reconstruct runner/provider and process indices.

    This function is executed on a Dask worker. It reads the shard spec JSON,
    imports the designated Runner/Provider classes, rebuilds the DictConfig and
    provider, constructs the per-worker environment, and applies
    ``RunnerClass.forward`` to each index in the shard.

    If ``result_path`` is provided, results are streamed to a JSONL file on the
    worker and ``None`` is returned; otherwise the list of results is returned
    to the driver.

    Args:
        spec_path (str): Path to the shard spec JSON file.

    Returns:
        list | None: List of results, or ``None`` if results were written to file.

    Raises:
        ImportError: If the runner/provider import path is invalid.
        Exception: Any exception thrown by environment builders or ``forward``.

    Notes:
        - Non-JSON-serializable results are ``repr``-serialized when writing JSONL.
    """
    from omegaconf import DictConfig, OmegaConf

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    RunnerCls = _import_obj(spec["runner_cls"])
    ProviderCls = _import_obj(spec["provider_cls"])

    cfg = DictConfig(spec["config"])
    params = spec.get("params", {}) or {}
    provider = ProviderCls(cfg, params=params)

    setup_fn = provider.make_worker_setup_fn()
    env = setup_fn()

    f = RunnerCls.forward  # staticmethod
    results = []
    for idx in spec["indices"]:
        results.append(f(idx, **env))

    result_path = spec.get("result_path")
    if result_path:
        with open(result_path, "w", encoding="utf-8") as w:
            for r in results:
                try:
                    json.dump(r, w, ensure_ascii=False)
                    w.write("\n")
                except TypeError:
                    w.write(repr(r) + "\n")
        return None

    return results
