"""BaseRunner class for orchestrating local, parallel, and async executions."""

import asyncio
import importlib
import json
import logging
import os
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from uuid import uuid4

from dask.utils import tmpfile
from omegaconf import OmegaConf
from tqdm import tqdm

from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.parallel.parallel import (
    build_client,
    get_client,
    get_parallel_config,
    wrap_func_with_worker_env,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsyncJobSpec:
    """Specification of an asynchronous shard submitted to the Dask cluster.

    Attributes:
        runner_cls (str): Import path to the concrete ``BaseRunner`` subclass.
        provider_cls (str): Import path to the concrete ``EnvironmentProvider`` subclass
        config (Dict): A resolved, plain-``dict`` serialization of the Hydra config.
        params (Dict): Extra parameters forwarded to the provider/environment.
        items (List[Any]): Pre-chunked items this shard should process.
        world_size (int): Total number of shards.
        world_rank (int): The shard rank (0..world_size-1).
        output_dir (str): Root output directory for shard-local outputs.
        shard_subdir (str): Optional sub-path under ``output_dir``.

    Notes:
        - Each shard is independent and reconstructs both provider and runner
          by import path on the worker before running the reducer pipeline.
    """

    runner_cls: str  # e.g., module.path.to.your.RunnerClass
    provider_cls: str  # e.g., module.path.to.your.ProviderClass
    config: Dict
    params: Dict
    items: List[Any]
    world_size: int
    world_rank: int
    output_dir: str
    shard_subdir: str


def _import_obj(cls_path: str):
    mod, _, name = cls_path.rpartition(".")
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


def get_job_class(cluster, spec_path=None):
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


def cat_shard_files(
    shard_dirs: Sequence[Path],
    relative_name: str,
    out_path: Path,
) -> bool:
    """Concatenate per-shard text files into one file."""
    found = False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        for shard_dir in shard_dirs:
            fragment = Path(shard_dir) / relative_name
            if not fragment.exists():
                continue
            found = True
            with fragment.open("r", encoding="utf-8") as rf:
                shutil.copyfileobj(rf, wf)
    if not found:
        out_path.unlink(missing_ok=True)
    return found


class BaseRunner(ABC):
    """A thin orchestration layer to run static ``forward`` over indices.

    This class handles:
        - Switching among local, parallel, and asynchronous (distributed) modes.
        - Injecting per-worker environments supplied by an :class:`EnvironmentProvider`.
        - Keeping ``forward`` as a ``@staticmethod`` for pickle-safety.
        - Optionally reducing per-shard results on the worker before they are
          merged on the driver.

    Subclass contract:
        - Implement ``@staticmethod forward(idx, dataset, model, **env) -> Any``
          without capturing ``self``. ``idx`` may be a single index or a batch
          of indices depending on ``batch_size``.
        - Provide an :class:`EnvironmentProvider` that builds the required env
          (e.g., dataset/model) for local and worker executions.
        - Override ``open_writers`` / ``write_record`` / ``close_writers`` /
          ``merge_scalar`` / ``merge_shard_files`` when reducer-style execution
          is needed.

    Args:
        provider (EnvironmentProvider): Provider that builds the runtime env.
        batch_size (int | None): If set, chunk indices into batches of this size
            before dispatching to ``forward``.
        async_mode (bool): If True, use Dask ``submit`` with asynchronous shards.
        async_specs_dir (str | Path): Output directory for per-shard spec JSON files.
        async_result_dir (str | Path): Output directory for per-shard async state.
        output_dir (str | Path | None): Root output directory used by reducer-style
            runners to write shard-local files and merged outputs.
        shard_subdir (str): Optional sub-path under ``output_dir`` or
            ``async_result_dir``.

    Notes:
        - In parallel sync mode (when a Dask cluster is configured), each worker
          returns a finalized shard state and the driver merges those states.
        - In async mode, each shard persists its finalized state on the worker.
    """

    # TODO(Masao) Add detailed description on Runner/Provider in the document.

    def __init__(
        self,
        provider: EnvironmentProvider,
        batch_size: int | None = None,
        async_mode: bool = False,
        async_specs_dir: str | Path = "./_async_specs",
        async_result_dir: str | Path = "./_async_results",
        output_dir: str | Path | None = None,
        shard_subdir: str = "",
    ):
        """Initialize BaseRunner object."""
        self.provider = provider
        self.batch_size = batch_size
        self.async_mode = async_mode
        self.async_specs_dir = Path(async_specs_dir).resolve()
        self.async_result_dir = Path(async_result_dir).resolve()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.shard_subdir = shard_subdir or ""

    @staticmethod
    @abstractmethod
    def forward(idx: int | Iterable[int], dataset, model, **env) -> Any:
        """Compute items for the given index or batch (to be implemented by subclasses).

        Keep this as a ``@staticmethod`` so that it is pickle-safe for Dask
        and does not capture ``self``.

        Args:
            idx (int | Iterable[int]): The input index or batch of indices to process.
            dataset: Dataset object provided via the environment.
            model: Model object provided via the environment.
            **env: Any additional environment entries injected by the provider.

        Returns:
            Any: Result for the given index or batch.

        Raises:
            NotImplementedError: Always in the base class; implement in subclass.
        """
        raise NotImplementedError

    @staticmethod
    def open_writers(shard_dir: Optional[Path], **env) -> Dict[str, Any]:
        """Open per-shard writers."""
        return {}

    @staticmethod
    def write_record(
        writers: Dict[str, Any],
        result: Any,
        state: Dict[str, Any],
        **env,
    ) -> None:
        """Persist one ``forward`` result."""
        state.setdefault("records", []).append(result)

    @staticmethod
    def close_writers(writers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Close per-shard writers."""
        for writer in writers.values():
            close = getattr(writer, "close", None)
            if callable(close):
                close()
        return None

    def merge_scalar(self, states: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Combine scalar accumulators on shard states."""
        return None

    def merge_shard_files(
        self,
        shard_dirs: List[Path],
        states: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Concatenate per-shard files into final outputs."""
        return None

    @staticmethod
    def serialize_state(state: Dict[str, Any], shard_dir: Path) -> Path:
        """Persist a finalized shard state."""
        import pickle

        path = Path(shard_dir) / "_state.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(state, f)
        return path

    @staticmethod
    def deserialize_state(shard_dir: Path) -> Dict[str, Any]:
        """Read a finalized shard state."""
        import pickle

        with (Path(shard_dir) / "_state.pkl").open("rb") as f:
            return pickle.load(f)

    @classmethod
    def init_state(
        cls,
        shard_id: int = 0,
        output_dir: Optional[str] = None,
        shard_subdir: str = "",
        **env,
    ) -> Dict[str, Any]:
        """Build the initial state for one shard."""
        shard_dir = None
        if output_dir is not None:
            shard_dir = Path(output_dir) / (shard_subdir or "") / f"split.{shard_id}"
            shard_dir.mkdir(parents=True, exist_ok=True)
        writers = cls.open_writers(
            shard_dir,
            shard_id=shard_id,
            output_dir=output_dir,
            shard_subdir=shard_subdir,
            **env,
        )
        return {
            "shard_id": shard_id,
            "shard_dir": str(shard_dir) if shard_dir is not None else None,
            "_writers": writers,
            "records": [],
        }

    @classmethod
    def reduce_state(cls, state: Dict[str, Any], result: Any, **env) -> Dict[str, Any]:
        """Fold a single ``forward`` result into the shard state."""
        cls.write_record(state["_writers"], result, state, **env)
        return state

    @classmethod
    def finalize_state(cls, state: Dict[str, Any], **env) -> Dict[str, Any]:
        """Close writers and finalize the shard state."""
        meta = cls.close_writers(state.get("_writers", {})) or {}
        if meta:
            state.update(meta)
        state.pop("_writers", None)
        return state

    def merge_states(self, states: List[Any]) -> Any:
        """Combine finalized shard states on the driver."""
        shard_dirs = [
            Path(state["shard_dir"])
            for state in states
            if isinstance(state, dict) and state.get("shard_dir")
        ]

        scalar = self.merge_scalar(states)
        files = self.merge_shard_files(shard_dirs, states) if shard_dirs else None

        if scalar is not None or files is not None:
            merged: Dict[str, Any] = {}
            if isinstance(scalar, dict):
                merged.update(scalar)
            if isinstance(files, dict):
                merged.update(files)
            return merged or (files if files is not None else scalar)

        out: List[Any] = []
        for state in states:
            if isinstance(state, dict):
                records = state.get("records")
                if records is not None:
                    out.extend(records)
            elif isinstance(state, list):
                out.extend(state)
            else:
                out.append(state)
        return out

    def _augment_env(self, env: Dict[str, Any]) -> Dict[str, Any]:
        if self.output_dir is not None:
            env.setdefault("output_dir", str(self.output_dir))
        if self.shard_subdir:
            env.setdefault("shard_subdir", self.shard_subdir)
        return env

    def _run_local(self, indices: Sequence[Any]) -> Any:
        """Run sequentially on the driver using a locally built environment.

        Args:
            indices (Sequence[Any]): Indices or batches to process.

        Returns:
            Any: Merged result for the single local shard.

        Notes:
            - Uses ``tqdm`` progress bar over the input sequence.
            - Reducer hooks run exactly once on the driver-side shard state.
        """
        env = self._augment_env(self.provider.build_env_local())
        cls = self.__class__

        state = cls.init_state(shard_id=0, **env)
        for item in tqdm(indices, total=len(indices)):
            result = cls.forward(item, **env)
            state = cls.reduce_state(state, result, shard_id=0, **env)
        state = cls.finalize_state(state, shard_id=0, **env)
        return self.merge_states([state])

    def _run_parallel(self, indices: Sequence[Any]) -> Any:
        """Run with Dask, one shard per worker, streaming finalized states.

        Args:
            indices (Sequence[Any]): Indices or batches to process.

        Returns:
            Any: Result of merging finalized shard states on the driver.

        Notes:
            - Wraps the shard task with :func:`wrap_func_with_worker_env` so that
              missing keyword args are injected from the worker env.
        """
        from dask.distributed import as_completed

        par_config = get_parallel_config()
        n_workers = int(par_config.get("n_workers", 1))
        shards = _default_chunk(list(indices), n_workers)
        if not shards:
            return self.merge_states([])

        shard_inputs = list(enumerate(shards))

        provider_setup = self.provider.build_worker_setup_fn()
        output_dir_str = str(self.output_dir) if self.output_dir is not None else None
        shard_subdir = self.shard_subdir

        def setup_fn():
            env = provider_setup()
            if output_dir_str is not None:
                env.setdefault("output_dir", output_dir_str)
            if shard_subdir:
                env.setdefault("shard_subdir", shard_subdir)
            return env

        runner_cls = self.__class__

        def shard_task(shard_input, **env):
            shard_id, items = shard_input
            state = runner_cls.init_state(shard_id=shard_id, **env)
            for item in items:
                result = runner_cls.forward(item, **env)
                state = runner_cls.reduce_state(state, result, shard_id=shard_id, **env)
            return runner_cls.finalize_state(state, shard_id=shard_id, **env)

        wrapped = wrap_func_with_worker_env(shard_task)
        states = []
        with get_client(par_config, setup_fn=setup_fn) as client:
            futures = client.map(wrapped, shard_inputs)
            try:
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="shards",
                ):
                    states.append(future.result())
            except Exception:
                try:
                    client.cancel(futures)
                finally:
                    raise
        return self.merge_states(states)

    async def _run_async(self, items: Sequence[Any]) -> List[Dict[str, Any]]:
        """Submit shards to a Dask cluster and persist shard states.

        Workflow:
            1) Emit per-shard JSON specs to ``async_specs_dir``.
            2) Submit worker entrypoints that reconstruct runner/provider and run.
            3) Write finalized shard states under ``output_dir`` when configured,
               otherwise under ``async_result_dir``.

        Args:
            items (Sequence[Any]): Indices or batches to process.

        Returns:
            List[Dict[str, Any]]: Submission metadata for the detached shard jobs.

        Raises:
            RuntimeError: If Dask client creation fails.
            ValueError: If async sharding parameters are invalid.
        """
        async_output_dir = (
            self.output_dir if self.output_dir is not None else self.async_result_dir
        )

        par_config = get_parallel_config()
        client = build_client(par_config)
        n_workers = par_config.get("n_workers", 1)
        try:
            chunks = _default_chunk(items, n_workers)
            self.async_specs_dir.mkdir(parents=True, exist_ok=True)

            config_dict = OmegaConf.to_container(self.provider.config, resolve=True)
            provider_cls = get_full_class_path_from_instance(self.provider)
            runner_cls = get_full_class_path_from_instance(self)

            job_meta = []
            for rank, chunk in enumerate(chunks):
                job_id = f"{uuid4().hex[:8]}-r{rank}"
                spec = AsyncJobSpec(
                    runner_cls=runner_cls,
                    provider_cls=provider_cls,
                    config=config_dict,
                    params=getattr(self.provider, "params", {}) or {},
                    items=list(chunk),
                    world_size=len(chunks),
                    world_rank=rank,
                    output_dir=str(async_output_dir),
                    shard_subdir=self.shard_subdir,
                )
                spec_path = self.async_specs_dir / f"spec-{job_id}.json"
                with open(spec_path, "w", encoding="utf-8") as f:
                    json.dump(convert_paths(asdict(spec)), f, ensure_ascii=False)

                client.cluster.job_cls = get_job_class(client.cluster, spec_path)

                with tmpfile(extension="sh") as tf:
                    with open(tf, "w", encoding="utf-8") as wtf:
                        wtf.write(client.cluster.job_script())

                    out = await client.cluster.job_cls._submit_job(
                        client.cluster.job_cls, tf
                    )
                    logger.info("Async job submission output: %s", out)

                shard_dir = async_output_dir / self.shard_subdir / f"split.{rank}"
                job_meta.append(
                    {
                        "job_id": job_id,
                        "spec": spec_path.resolve().as_posix(),
                        "shard_dir": str(shard_dir),
                    }
                )

            logger.info(
                "Detached async submission. Scheduler: %s",
                getattr(client, "scheduler_info", lambda: {})().get("address", "?"),
            )
            return job_meta

        finally:
            client.close()

    def merge_async_results(self) -> Any:
        """Combine shard states written by an async run."""
        base_dir = self.output_dir if self.output_dir is not None else self.async_result_dir
        base = base_dir / self.shard_subdir
        shard_dirs = sorted(
            (p for p in base.glob("split.*") if (p / "_state.pkl").exists()),
            key=lambda p: int(p.name.split(".", 1)[1]),
        )
        if not shard_dirs:
            raise FileNotFoundError(
                f"No shard states found under {base}; have async jobs completed?"
            )
        cls = self.__class__
        states = [cls.deserialize_state(d) for d in shard_dirs]
        return self.merge_states(states)

    def __call__(self, indices: Iterable[int]) -> Any:
        """Dispatch execution according to the configured parallel mode.

        Args:
            indices (Iterable[int]): Indices to process.

        Returns:
            Any: Results for local/parallel modes, or async submission metadata
            in async mode.

        Notes:
            - If no parallel config is set or ``env='local'``, run locally.
            - Otherwise, run in parallel with environment injection.
        """
        indices = list(indices)
        if self.batch_size is not None:
            if self.batch_size <= 0:
                raise ValueError("batch_size must be a positive integer.")
            indices = _chunk_indices(indices, self.batch_size)
        if self.async_mode:
            return asyncio.run(self._run_async(indices))

        par_config = get_parallel_config()
        if par_config is None or getattr(par_config, "env", "local") == "local":
            return self._run_local(indices)
        return self._run_parallel(indices)


def _chunk_indices(indices: Sequence[int], batch_size: int) -> List[List[int]]:
    """Split a sequence of indices into fixed-size chunks."""
    batches = []
    for i in range(0, len(indices), batch_size):
        batches.append(list(indices[i : i + batch_size]))
    return [b for b in batches if b]


def _async_worker_entry_from_spec_path(spec_path: str) -> None:
    """Worker entrypoint: reconstruct runner/provider and process one shard.

    This function is executed on a Dask worker. It reads the shard spec JSON,
    imports the designated Runner/Provider classes, rebuilds the DictConfig and
    provider, constructs the per-worker environment, and runs the reducer
    pipeline for every item in the shard.

    Args:
        spec_path (str): Path to the shard spec JSON file.

    Returns:
        None: The finalized shard state is serialized under the shard directory.

    Raises:
        ImportError: If the runner/provider import path is invalid.
        Exception: Any exception thrown by environment builders or ``forward``.

    Notes:
        - The finalized shard state is written via
          ``RunnerCls.serialize_state(...)`` and is later consumed by
          ``BaseRunner.merge_async_results`` on the driver.
    """
    from omegaconf import DictConfig

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    RunnerCls = _import_obj(spec["runner_cls"])
    ProviderCls = _import_obj(spec["provider_cls"])

    config = DictConfig(spec["config"])
    params = spec.get("params", {}) or {}
    provider = ProviderCls(config, params=params)

    # Add world size and rank
    os.environ["WORLD_SIZE"] = str(spec["world_size"])
    os.environ["WORLD_RANK"] = str(spec["world_rank"])

    setup_fn = provider.build_worker_setup_fn()
    env = setup_fn()

    output_dir = spec.get("output_dir")
    shard_subdir = spec.get("shard_subdir", "") or ""
    if output_dir is not None:
        env.setdefault("output_dir", output_dir)
    if shard_subdir:
        env.setdefault("shard_subdir", shard_subdir)

    shard_id = int(spec["world_rank"])
    state = RunnerCls.init_state(shard_id=shard_id, **env)
    for item in spec["items"]:
        result = RunnerCls.forward(item, **env)
        state = RunnerCls.reduce_state(state, result, shard_id=shard_id, **env)
    state = RunnerCls.finalize_state(state, shard_id=shard_id, **env)

    shard_dir = (
        Path(state["shard_dir"])
        if state.get("shard_dir")
        else Path(output_dir or ".") / shard_subdir / f"split.{shard_id}"
    )
    RunnerCls.serialize_state(state, shard_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=str, help="Path to AsyncJobSpec JSON file")
    args = parser.parse_args()
    _async_worker_entry_from_spec_path(args.spec)
