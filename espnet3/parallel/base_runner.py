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
        items (List[Any]): Pre-chunked items this shard should process. Each
            entry is either an integer index or a list of indices when the
            runner uses ``batch_size``.
        world_size (int): Total number of shards.
        world_rank (int): The shard rank (0..world_size-1).
        output_dir (str): Root output directory. Each shard writes into
            ``<output_dir>/<shard_subdir>/split.<world_rank>/``.
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
    """Concatenate per-shard text files into a single output file.

    Each shard directory is expected to contain a file named ``relative_name``
    (e.g. ``"hyp0.scp"``). They are concatenated in shard order into
    ``out_path``. Missing fragments are skipped silently.

    Args:
        shard_dirs: Per-shard directories produced by the runner.
        relative_name: Path of the fragment relative to each shard dir.
        out_path: Destination file. Parent directories are created as needed.

    Returns:
        ``True`` if at least one fragment was found and written.
    """
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
    """A thin orchestration layer to run ``forward`` over indices with reducers.

    Each shard is processed on a single worker (or the driver in local mode).
    Items are folded through a small set of overridable hooks so that heavy
    payloads such as features or audio can be persisted directly on the worker
    side, and only small summaries travel back to the driver.

    Default behaviour preserves the legacy ``List[Any]`` return type so
    subclasses that only implement ``forward`` keep working unchanged.

    Persistence hooks (override on the subclass):
        - :py:meth:`open_writers` — open per-shard writers for ``split.<rank>/``.
        - :py:meth:`write_record` — fold one ``forward`` result into the writers
          and/or into scalar accumulators on the state.
        - :py:meth:`close_writers` — close writers and optionally return metadata
          to attach to the shard state.
        - :py:meth:`merge_scalar` — combine scalar accumulators across shards.
        - :py:meth:`merge_shard_files` — concatenate per-shard files into the
          final output paths.

    Args:
        provider (EnvironmentProvider): Provider that builds the runtime env.
        batch_size (int | None): If set, chunk indices into batches of this size
            before dispatching to ``forward``.
        async_mode (bool): If True, submit detached batch jobs (e.g., SLURM)
            instead of awaiting results in the driver. Each job runs the
            reducer pipeline and persists its finalized state to disk; the
            driver returns submission metadata immediately. Use
            :py:meth:`merge_async_results` to combine the shard states after
            every job has completed.
        async_specs_dir (str | Path): Output directory for per-shard spec JSON
            files (only used in async mode).
        output_dir (str | Path | None): When provided, each shard receives a
            dedicated ``<output_dir>/<shard_subdir>/split.<rank>/`` directory.
            If ``None``, the runner falls back to the in-memory list mode.
            Required in async mode (workers need a place to persist state).
        shard_subdir (str): Optional sub-path under ``output_dir`` (e.g. the
            split name or test set name).

    Notes:
        - In parallel mode, indices are split into ``n_workers`` shards. Each
          shard task folds its items through ``reduce_state`` on the worker and
          returns the finalized state; the driver consumes states as they
          complete (``as_completed``) and calls :py:meth:`merge_states`.
        - In async mode, the same reducer pipeline runs on the worker side,
          but the driver detaches after submission. Each worker writes its
          finalized state via :py:meth:`serialize_state` so a later
          :py:meth:`merge_async_results` call can combine them.
    """

    def __init__(
        self,
        provider: EnvironmentProvider,
        batch_size: int | None = None,
        async_mode: bool = False,
        async_specs_dir: str | Path = "./_async_specs",
        output_dir: str | Path | None = None,
        shard_subdir: str = "",
    ):
        """Initialize BaseRunner object."""
        self.provider = provider
        self.batch_size = batch_size
        self.async_mode = async_mode
        self.async_specs_dir = Path(async_specs_dir).resolve()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.shard_subdir = shard_subdir or ""

    # ------------------------------------------------------------------
    # Abstract / overridable hooks
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def forward(idx: int | Iterable[int], dataset, model, **env) -> Any:
        """Compute the result for a single item (or batch of items)."""
        raise NotImplementedError

    @staticmethod
    def open_writers(shard_dir: Optional[Path], **env) -> Dict[str, Any]:
        """Open per-shard writers.

        Called once per shard on the worker. ``shard_dir`` is the unique
        ``split.<rank>/`` directory if ``output_dir`` was configured on the
        runner, otherwise ``None``. The returned dict is passed back to
        :py:meth:`write_record` and :py:meth:`close_writers` and is discarded
        before the state ships to the driver (writers are not picklable).

        Default: no writers (in-memory list mode).
        """
        return {}

    @staticmethod
    def write_record(
        writers: Dict[str, Any],
        result: Any,
        state: Dict[str, Any],
        **env,
    ) -> None:
        """Persist one ``forward`` result.

        Subclasses can use ``writers`` to stream the result to disk and/or
        mutate ``state`` to fold the result into scalar accumulators. The
        default keeps ``result`` in memory under ``state['records']`` so that
        subclasses that only implement ``forward`` see the legacy
        ``List[Any]`` return semantics.
        """
        state.setdefault("records", []).append(result)

    @staticmethod
    def close_writers(writers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Close per-shard writers.

        Default closes every value that exposes a ``.close()`` method.
        Subclasses can return a dict whose keys are merged into the shard
        state before it is sent to the driver (useful for passing along
        bookkeeping such as the keys that were actually written).
        """
        for writer in writers.values():
            close = getattr(writer, "close", None)
            if callable(close):
                close()
        return None

    def merge_scalar(self, states: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Combine scalar accumulators carried on shard states.

        Default: ``None`` (no scalar reduction performed).
        """
        return None

    def merge_shard_files(
        self,
        shard_dirs: List[Path],
        states: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Concatenate per-shard files into the final output paths.

        Default: ``None`` (no file merging performed). Subclasses can use
        :func:`cat_shard_files` for typical scp/shape concatenation.
        """
        return None

    @staticmethod
    def serialize_state(state: Dict[str, Any], shard_dir: Path) -> Path:
        """Persist a finalized shard state for later driver-side merge.

        Used by async mode: each worker writes its state to disk after the
        reducer pipeline finishes, so a later :py:meth:`merge_async_results`
        call can read them back without re-running anything. Default uses
        pickle so numpy/torch payloads survive the round-trip; override to
        switch formats.
        """
        import pickle

        path = Path(shard_dir) / "_state.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(state, f)
        return path

    @staticmethod
    def deserialize_state(shard_dir: Path) -> Dict[str, Any]:
        """Read a finalized shard state previously written to ``shard_dir``."""
        import pickle

        with (Path(shard_dir) / "_state.pkl").open("rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Default reducer plumbing built on top of the hooks above
    # ------------------------------------------------------------------

    @classmethod
    def init_state(
        cls,
        shard_id: int = 0,
        output_dir: Optional[str] = None,
        shard_subdir: str = "",
        **env,
    ) -> Dict[str, Any]:
        """Build the initial state for one shard.

        Creates the ``split.<rank>/`` directory if ``output_dir`` was supplied
        and asks the subclass for any per-shard writers.
        """
        shard_dir: Optional[Path] = None
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
        """Close writers and strip non-picklable entries before the trip back."""
        meta = cls.close_writers(state.get("_writers", {})) or {}
        if meta:
            state.update(meta)
        state.pop("_writers", None)
        return state

    def merge_states(self, states: List[Any]) -> Any:
        """Combine per-shard finalized states on the driver.

        - Calls :py:meth:`merge_scalar` and :py:meth:`merge_shard_files`. If
          either returns a non-``None`` dict the merged dict is returned.
        - Otherwise falls back to flattening ``state['records']`` from every
          shard (legacy list semantics).
        """
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

    # ------------------------------------------------------------------
    # Execution paths
    # ------------------------------------------------------------------

    def _augment_env(self, env: Dict[str, Any]) -> Dict[str, Any]:
        if self.output_dir is not None:
            env.setdefault("output_dir", str(self.output_dir))
        if self.shard_subdir:
            env.setdefault("shard_subdir", self.shard_subdir)
        return env

    def _run_local(self, indices: Sequence[Any]) -> Any:
        """Run sequentially on the driver as a single shard."""
        env = self._augment_env(self.provider.build_env_local())
        cls = self.__class__

        state = cls.init_state(shard_id=0, **env)
        for item in tqdm(indices, total=len(indices)):
            result = cls.forward(item, **env)
            state = cls.reduce_state(state, result, shard_id=0, **env)
        state = cls.finalize_state(state, shard_id=0, **env)
        return self.merge_states([state])

    def _run_parallel(self, indices: Sequence[Any]) -> Any:
        """Run with Dask, one shard per worker, streaming finalized states."""
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
        states: List[Any] = []
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
        """Submit detached per-shard batch jobs and return submission metadata.

        Each job runs the reducer pipeline on its shard on the worker side and
        persists the finalized state into
        ``<output_dir>/<shard_subdir>/split.<rank>/_state.pkl`` via
        :py:meth:`serialize_state`. The driver does not gather; once all jobs
        complete the user calls :py:meth:`merge_async_results` to combine
        them.

        Args:
            items: Pre-chunked items to process (one entry per ``forward``
                invocation; may be ints or lists of ints when ``batch_size``
                is set).

        Returns:
            List of submission metadata dicts (``job_id``, ``spec``,
            ``shard_dir``).

        Raises:
            ValueError: If ``output_dir`` is not configured on the runner.
        """
        if self.output_dir is None:
            raise ValueError(
                "async_mode requires output_dir on the runner; workers need a "
                "place to persist per-shard state."
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

            job_meta: List[Dict[str, Any]] = []
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
                    output_dir=str(self.output_dir),
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

                shard_dir = self.output_dir / self.shard_subdir / f"split.{rank}"
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
        """Combine per-shard states written by an async run.

        Reads every ``<output_dir>/<shard_subdir>/split.*/_state.pkl`` (in
        rank order) via :py:meth:`deserialize_state` and feeds them to
        :py:meth:`merge_states`.

        Returns:
            Whatever :py:meth:`merge_states` returns for the runner.

        Raises:
            ValueError: If ``output_dir`` is not configured on the runner.
            FileNotFoundError: If no shard states are found at the expected
                location (typically means the async jobs have not completed
                yet, or were never submitted).
        """
        if self.output_dir is None:
            raise ValueError("merge_async_results requires output_dir on the runner.")
        base = self.output_dir / self.shard_subdir
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
        """Dispatch execution according to the configured parallel mode."""
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
    """Worker entrypoint: rebuild the runner and persist the shard state.

    Executed inside the detached batch job. It rehydrates the runner and
    provider from the spec, runs the same reducer pipeline used by sync
    parallel mode on this shard's items, and writes the finalized state to
    ``<output_dir>/<shard_subdir>/split.<rank>/_state.pkl`` via
    :py:meth:`BaseRunner.serialize_state`. A subsequent
    :py:meth:`BaseRunner.merge_async_results` call on the driver reads those
    files back and produces the final merged result.
    """
    from omegaconf import DictConfig

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    RunnerCls = _import_obj(spec["runner_cls"])
    ProviderCls = _import_obj(spec["provider_cls"])

    config = DictConfig(spec["config"])
    params = spec.get("params", {}) or {}
    provider = ProviderCls(config, params=params)

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
