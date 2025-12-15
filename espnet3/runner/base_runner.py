"""BaseRunner class for orchestrating local, parallel, and async executions."""

import asyncio
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from uuid import uuid4

from dask.utils import tmpfile
from omegaconf import OmegaConf
from tqdm import tqdm

from espnet3.parallel.parallel import (
    get_client,
    get_parallel_config,
    make_client,
    parallel_for,
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
    return [c for c in out if c]


def convert_paths(obj):
    """Recursively convert Path objects to strings in the given object."""
    if isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def get_full_class_path_from_instance(obj):
    """Return the full import path of the given object's class."""
    cls = obj.__class__
    module = cls.__module__

    # replace top actual file name if __main__
    if module == "__main__":
        import __main__

        filename = os.path.splitext(os.path.basename(__main__.__file__))[0]
        module = filename

    return f"{module}.{cls.__qualname__}"


def get_job_cls(cluster, spec_path=None):
    """Dask Job class that submits async runner jobs with the given spec path."""
    parent_cls = cluster.job_cls
    assert spec_path is not None

    class ASyncRunnerJob(parent_cls):
        def __init__(self, *args, worker_extra_args=None, **kwargs):
            self._user_worker_extra_args = worker_extra_args or []
            super().__init__(*args, worker_extra_args=worker_extra_args, **kwargs)
            python = sys.executable
            current_file_path = Path(__file__).resolve()

            # Update command template to submit async parallel jobs with Dask
            self._command_template = f"{python} {current_file_path} {spec_path} "

    return ASyncRunnerJob


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
        provider (EnvironmentProvider): Provider that builds the runtime env.
        async_mode (bool): If True, use Dask ``submit`` with asynchronous shards.
        async_specs_dir (str | Path): Output directory for per-shard spec JSON files.
        async_num_workers (int | None): If set, overrides detected worker count
            to decide how many shards to create.
        async_result_dir (str | Path): Output directory for per-shard JSONL results.

    Notes:
        - In parallel sync mode (when a Dask cluster is configured), tasks are
          submitted via ``parallel_map`` and results are gathered in order.
        - In async mode, the results will be written on async_result_dir.
    """

    # TODO(Masao) Add detailed description on Runner/Provider in the document.

    def __init__(
        self,
        provider: EnvironmentProvider,
        *,
        async_mode: bool = False,
        async_specs_dir: str | Path = "./_async_specs",
        async_result_dir: str | Path = "./_async_results",
    ):
        """Initialize BaseRunner object."""
        self.provider = provider
        self.async_mode = async_mode
        self.async_specs_dir = Path(async_specs_dir).resolve()
        self.async_result_dir = Path(async_result_dir).resolve()

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
        env = self.provider.build_env_local()
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
        setup_fn = self.provider.make_worker_setup_fn()
        out = []
        with get_client(get_parallel_config()) as client:
            for res in tqdm(
                parallel_for(
                    self.__class__.forward, indices, setup_fn=setup_fn, client=client
                ),
                total=len(indices),
            ):
                out.append(res)
        return out

    async def _run_async(self, indices: Sequence[int]) -> List[Any] | None:
        """Submit shards to a Dask cluster and gather or write results.

        Workflow:
            1) Emit per-shard JSON specs to ``async_specs_dir``.
            2) Submit worker entrypoints that reconstruct runner/provider and run.
            3) Write results to ``async_result_dir``.
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
        client = make_client(par_cfg)
        n_workers = par_cfg.get("n_workers", 1)
        try:
            chunks = _default_chunk(indices, n_workers)
            self.async_specs_dir.mkdir(parents=True, exist_ok=True)
            self.async_result_dir.mkdir(parents=True, exist_ok=True)

            # DictConfig -> dict
            cfg_dict = OmegaConf.to_container(self.provider.config, resolve=True)
            provider_cls = get_full_class_path_from_instance(self.provider)
            runner_cls = get_full_class_path_from_instance(self)

            job_meta = []
            for rank, chunk in enumerate(chunks):
                job_id = f"{uuid4().hex[:8]}-r{rank}"
                result_path = (
                    self.async_result_dir / f"result-{job_id}.jsonl"
                ).as_posix()

                spec = AsyncJobSpec(
                    runner_cls=runner_cls,
                    provider_cls=provider_cls,
                    config=cfg_dict,
                    params=getattr(self.provider, "params", {}) or {},
                    indices=list(chunk),
                    world_size=len(chunks),
                    world_rank=rank,
                    result_path=str(result_path),
                    extras={},
                )
                spec_path = self.async_specs_dir / f"spec-{job_id}.json"
                with open(spec_path, "w", encoding="utf-8") as f:
                    json.dump(convert_paths(asdict(spec)), f, ensure_ascii=False)

                client.cluster.job_cls = get_job_cls(client.cluster, spec_path)

                with tmpfile(extension="sh") as tf:
                    with open(tf, "w", encoding="utf-8") as wtf:
                        wtf.write(client.cluster.job_script())

                    out = await client.cluster.job_cls._submit_job(
                        client.cluster.job_cls, tf
                    )
                    print(out)  # Print job submission output.

                job_meta.append(
                    {
                        "job_id": job_id,
                        "spec": spec_path.resolve().as_posix(),
                        "result": result_path,
                    }
                )

            print(
                "Detached async submission. Scheduler:",
                getattr(client, "scheduler_info", lambda: {})().get("address", "?"),
            )
            return job_meta

        finally:
            client.close()

    def __call__(self, indices: Iterable[int]) -> List[Any] | None:
        """Dispatch execution according to the configured parallel mode.

        Args:
            indices (Iterable[int]): Indices to process.

        Returns:
            List[Any] | None: Results for local/parallel modes, or ``None`` in
            async mode.

        Notes:
            - If no parallel config is set or ``env='local'``, run locally.
            - Otherwise, run in parallel with environment injection.
        """
        indices = list(indices)
        if self.async_mode:
            return asyncio.run(self._run_async(indices))

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
    from omegaconf import DictConfig

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    RunnerCls = _import_obj(spec["runner_cls"])
    ProviderCls = _import_obj(spec["provider_cls"])

    cfg = DictConfig(spec["config"])
    params = spec.get("params", {}) or {}
    provider = ProviderCls(cfg, params=params)

    # Add world size and rank
    os.environ["WORLD_SIZE"] = str(spec["world_size"])
    os.environ["WORLD_RANK"] = str(spec["world_rank"])

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=str, help="Path to AsyncJobSpec JSON file")
    args = parser.parse_args()
    _async_worker_entry_from_spec_path(args.spec)
