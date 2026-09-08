"""BaseRunner class for orchestrating local and parallel shard executions."""

import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tqdm import tqdm

from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.parallel.parallel import (
    get_client,
    get_parallel_config,
    wrap_func_with_worker_env,
)

logger = logging.getLogger(__name__)


def concatenate_shard_files(
    shard_dirs: Sequence[Path],
    relative_name: str,
    out_path: Path,
) -> bool:
    """Concatenate shard-local text files into one output file."""
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

    Subclass contract:
        - Implement ``@staticmethod forward(idx, dataset, model, **env) -> Any``
          without capturing ``self``. ``idx`` may be a single index or a batch
          of indices depending on ``batch_size``.
        - Provide an :class:`EnvironmentProvider` that builds the required env
          (e.g., dataset/model) for local and worker executions.

    Args:
        provider (EnvironmentProvider): Provider that builds the runtime env.
        batch_size (int | None): If set, chunk indices into batches of this size
            before dispatching to ``forward``.
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
        batch_size: int | None = None,
        output_dir: str | Path | None = None,
        shard_subdir: str = "",
        resume: bool = True,
    ):
        """Initialize BaseRunner object."""
        self.provider = provider
        self.batch_size = batch_size
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.shard_subdir = shard_subdir or ""
        self.resume = resume

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

        Example:
            >>> class MyRunner(BaseRunner):
            ...     @staticmethod
            ...     def forward(idx, dataset, model, **env):
            ...         if isinstance(idx, int):
            ...             x = dataset[idx]
            ...             return model(x)
            ...         xs = [dataset[i] for i in idx]
            ...         return model(xs)
        """
        raise NotImplementedError

    @staticmethod
    def open_writers(shard_dir: Optional[Path], **env) -> Dict[str, Any]:
        """Open per-shard writers before processing begins."""
        return {}

    @staticmethod
    def write_record(
        writers: Dict[str, Any],
        result: Any,
        state: Dict[str, Any],
        **env,
    ) -> None:
        """Persist one ``forward`` result into the shard state or files."""
        state.setdefault("records", []).append(result)

    @staticmethod
    def close_writers(
        writers: Dict[str, Any],
        state: Dict[str, Any],
        **env,
    ) -> Optional[Dict[str, Any]]:
        """Close per-shard writers after all items are processed."""
        for writer in writers.values():
            close = getattr(writer, "close", None)
            if callable(close):
                close()
        return None

    def merge(self, shard_dirs: List[Path]) -> Any:
        """Merge completed shard outputs into the final result."""
        return None

    @staticmethod
    def _get_shards_root(output_dir: Path, shard_subdir: str = "") -> Path:
        """Return the root directory for all shard subdirectories under output_dir."""
        root = Path(output_dir)
        if shard_subdir:
            root = root / shard_subdir
        return root

    @classmethod
    def _get_manifest_path(cls, output_dir: Path, shard_subdir: str = "") -> Path:
        """Return the manifest path for the shard plan."""
        return cls._get_shards_root(output_dir, shard_subdir) / "manifest.json"

    @classmethod
    def _resolve_shard_dir(
        cls,
        output_dir: str,
        shard_subdir: str,
        shard_id: int,
    ) -> Path:
        """Return the working directory for one shard (split.N)."""
        root = cls._get_shards_root(Path(output_dir), shard_subdir)
        return root / f"split.{shard_id}"

    @staticmethod
    def _get_done_path(shard_dir: Path) -> Path:
        """Return the sentinel file path that marks a shard as complete."""
        return Path(shard_dir) / "done"

    @classmethod
    def _get_lock_path(cls, shard_dir: Path) -> Path:
        """Return the sentinel file path that marks a shard as locked."""
        return Path(shard_dir) / "lock"

    @classmethod
    def is_shard_done(cls, shard_dir: Path) -> bool:
        """Return True if the shard has a completion marker file."""
        return cls._get_done_path(shard_dir).exists()

    @classmethod
    def _try_lock_shard(cls, shard_dir: Path) -> bool:
        """Create a shard lock file atomically and return True on success."""
        shard_dir = Path(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
        lock_path = cls._get_lock_path(shard_dir)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
            lock_file.write(f"{os.getpid()}\n")
        return True

    @classmethod
    def _unlock_shard(cls, shard_dir: Path) -> None:
        """Remove the shard lock file after processing finishes."""
        cls._get_lock_path(shard_dir).unlink(missing_ok=True)

    @classmethod
    def init_state(
        cls,
        shard_id: int = 0,
        output_dir: str = "",
        shard_subdir: str = "",
        **env,
    ) -> Dict[str, Any]:
        """Build the initial state dict for one shard and open its writers."""
        shard_dir = cls._resolve_shard_dir(output_dir, shard_subdir, shard_id)
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
            "shard_dir": str(shard_dir),
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
        meta = cls.close_writers(state.get("_writers", {}), state, **env) or {}
        if meta:
            state.update(meta)
        state.pop("_writers", None)
        return state

    def _write_manifest(self, shards: Sequence[Dict[str, Any]]) -> Path:
        """Write shard plan to manifest.json before dispatch in __call__."""
        manifest_path = self._get_manifest_path(self.output_dir, self.shard_subdir)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "output_dir": str(self.output_dir),
            "shard_subdir": self.shard_subdir,
            "shards": list(shards),
        }
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return manifest_path

    def _load_manifest(self) -> Optional[Dict[str, Any]]:
        """Load the existing shard manifest when resume is enabled."""
        manifest_path = self._get_manifest_path(self.output_dir, self.shard_subdir)
        if not manifest_path.exists():
            return None
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict):
            raise RuntimeError(f"Invalid shard manifest format: {manifest_path}")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise RuntimeError(f"Shard manifest is missing 'shards': {manifest_path}")
        return manifest

    def _plan_shards(self, items: Sequence[Any]) -> List[Dict[str, Any]]:
        """Divide items into per-shard specs per the active parallel config."""
        par_config = get_parallel_config()
        env = getattr(par_config, "env", "local") if par_config is not None else "local"
        num_shards = 1
        if par_config is not None and env not in ("local",):
            num_shards = int(getattr(par_config, "n_workers", 1))
        n_chunks = max(1, num_shards)
        items_list = list(items)
        quotient, remainder = divmod(len(items_list), n_chunks)
        chunks = []
        start = 0
        for i in range(n_chunks):
            size = quotient + (1 if i < remainder else 0)
            chunk = items_list[start : start + size]
            if chunk:
                chunks.append(chunk)
            start += size
        return [
            {"shard_id": shard_id, "items": chunk}
            for shard_id, chunk in enumerate(chunks)
        ]

    def _resolve_shards(self, items: Sequence[Any]) -> List[Dict[str, Any]]:
        """Resolve the shard plan for a fresh run or a resumed run."""
        planned_shards = self._plan_shards(items)
        if not self.resume:
            self._write_manifest(planned_shards)
            return planned_shards

        manifest = self._load_manifest()
        if manifest is None:
            self._write_manifest(planned_shards)
            return planned_shards

        manifest_shards = manifest["shards"]
        if len(manifest_shards) != len(planned_shards):
            raise RuntimeError(
                "Cannot resume with a different number of parallel shards. "
                f"Existing run has {len(manifest_shards)} shard(s), "
                f"but this run planned {len(planned_shards)}. "
                "Re-run with the original parallel setting or remove the "
                "existing shard outputs before starting over."
            )
        if manifest_shards != planned_shards:
            raise RuntimeError(
                "Cannot resume because the shard plan changed. "
                "The existing manifest does not match the current indices or "
                "batching configuration. Re-run with the original settings or "
                "remove the existing shard outputs before starting over."
            )
        return manifest_shards

    def _filter_pending_shards(
        self, shards: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return locked shards; skips done-marked ones when resume=True."""
        pending = []
        for shard in shards:
            shard_dir = self._resolve_shard_dir(
                str(self.output_dir), self.shard_subdir, int(shard["shard_id"])
            )
            if self.resume and self.is_shard_done(shard_dir):
                continue
            if not self._try_lock_shard(shard_dir):
                if self.resume and self.is_shard_done(shard_dir):
                    continue
                raise RuntimeError(
                    "Shard is already locked by another runner: " f"{shard_dir}"
                )
            if self.resume and self.is_shard_done(shard_dir):
                self._unlock_shard(shard_dir)
                continue
            pending.append(shard)
        return pending

    def _get_completed_shard_dirs(self, shards: Sequence[Dict[str, Any]]) -> List[Path]:
        """Return completed shard paths; raises if any done marker is absent."""
        shard_dirs = []
        for shard in shards:
            shard_dir = self._resolve_shard_dir(
                str(self.output_dir),
                self.shard_subdir,
                int(shard["shard_id"]),
            )
            if not self.is_shard_done(shard_dir):
                raise FileNotFoundError(
                    f"Shard {shard['shard_id']} is not complete: {shard_dir}"
                )
            shard_dirs.append(shard_dir)
        return shard_dirs

    @classmethod
    def _run_one_shard(
        cls,
        shard_id: int,
        items: Sequence[Any],
        env: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process one shard end-to-end; used by _run_local and _run_parallel_dask."""
        state = cls.init_state(shard_id=shard_id, **env)
        shard_dir = Path(state["shard_dir"])
        try:
            cls._get_done_path(shard_dir).unlink(missing_ok=True)
            for item in items:
                result = cls.forward(item, **env)
                state = cls.reduce_state(state, result, shard_id=shard_id, **env)
            cls.finalize_state(state, shard_id=shard_id, **env)
            cls._get_done_path(shard_dir).write_text("", encoding="utf-8")
            return state
        finally:
            cls._unlock_shard(shard_dir)

    def _run_local(self, shards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run pending shards sequentially on the driver."""
        env = self.provider.build_env_local()
        env.setdefault("output_dir", str(self.output_dir))
        if self.shard_subdir:
            env.setdefault("shard_subdir", self.shard_subdir)
        cls = self.__class__
        states = []
        for shard in tqdm(shards, total=len(shards), desc="shards"):
            states.append(
                cls._run_one_shard(int(shard["shard_id"]), shard["items"], env)
            )
        return states

    def _run_parallel_dask(
        self, shards: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run pending shards with Dask workers and persist shard-local state."""
        from dask.distributed import as_completed

        par_config = get_parallel_config()
        if not shards:
            return []

        provider_setup = self.provider.build_worker_setup_fn()
        output_dir_str = str(self.output_dir)
        shard_subdir = self.shard_subdir

        def setup_fn():
            env = provider_setup()
            env.setdefault("output_dir", output_dir_str)
            if shard_subdir:
                env.setdefault("shard_subdir", shard_subdir)
            return env

        runner_cls = self.__class__

        def shard_task(shard_spec, **env):
            return runner_cls._run_one_shard(
                int(shard_spec["shard_id"]),
                shard_spec["items"],
                env,
            )

        wrapped = wrap_func_with_worker_env(shard_task)
        states = []
        with get_client(par_config, setup_fn=setup_fn) as client:
            futures = client.map(wrapped, list(shards))
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
        return states

    def __call__(self, indices: Iterable[int]) -> Any:
        """Dispatch execution according to the configured parallel mode."""
        if self.output_dir is None:
            raise RuntimeError("BaseRunner requires output_dir for shard execution.")
        indices = list(indices)
        if self.batch_size is not None:
            if self.batch_size <= 0:
                raise ValueError("batch_size must be a positive integer.")
            indices = [
                list(indices[i : i + self.batch_size])
                for i in range(0, len(indices), self.batch_size)
            ]
        shards = self._resolve_shards(indices)
        pending = self._filter_pending_shards(shards)

        par_config = get_parallel_config()
        if pending:
            if par_config is None or getattr(par_config, "env", "local") == "local":
                self._run_local(pending)
            else:
                self._run_parallel_dask(pending)

        return self.merge(self._get_completed_shard_dirs(shards))
