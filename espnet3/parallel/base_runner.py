"""BaseRunner class for orchestrating local and parallel shard executions."""

import json
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from omegaconf import OmegaConf
from tqdm import tqdm

from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.parallel.parallel import (
    get_client,
    get_parallel_config,
    wrap_func_with_worker_env,
)

logger = logging.getLogger(__name__)


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


def _convert_paths(obj):
    """Recursively convert Path objects to strings in the given object."""
    if isinstance(obj, dict):
        return {k: _convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def cat_shard_files(
    shard_dirs: Sequence[Path],
    relative_name: str,
    out_path: Path,
) -> bool:
    """Concatenate shard-local text files into a single output file.

    Looks for ``relative_name`` under each directory in ``shard_dirs`` and
    appends the contents of every existing fragment to ``out_path`` in the
    given order. If no fragment exists, this function removes ``out_path`` when
    present and returns ``False``.

    Args:
        shard_dirs: Ordered shard directories to scan for file fragments.
        relative_name: Relative path of the fragment file inside each shard
            directory.
        out_path: Destination path for the concatenated file.

    Returns:
        bool: ``True`` if at least one shard fragment was found and written.
        Otherwise ``False``.
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
    """A thin orchestration layer to run static ``forward`` over shard items.

    This class handles:
        - Switching among local and parallel execution modes.
        - Injecting per-worker environments supplied by an :class:`EnvironmentProvider`.
        - Keeping ``forward`` as a ``@staticmethod`` for pickle-safety.
        - Persisting shard-local state so interrupted runs can resume.
        - Optionally reducing per-shard results on the worker before they are
          merged on the driver.

    Unlike ``parallel_map`` and ``parallel_for``, this class is intended for
    durable shard-oriented workflows that write files under ``output_dir`` and
    may need resume support. When ``output_dir`` is omitted, it still supports
    in-memory reduction for lightweight callers, but resumable execution is
    only enabled for file-backed runs.

    Subclass contract:
        - Implement ``@staticmethod forward(idx, dataset, model, **env) -> Any``
          without capturing ``self``. ``idx`` may be a single index or a batch
          of indices depending on ``batch_size``.
        - Provide an :class:`EnvironmentProvider` that builds the required env
          (e.g., dataset/model) for local and worker executions.
        - Override ``open_writers`` / ``write_record`` / ``close_writers`` /
          ``merge_state`` / ``merge_shard_files`` when reducer-style execution
          is needed.

    Args:
        provider (EnvironmentProvider): Provider that builds the runtime env.
        batch_size (int | None): If set, chunk indices into batches of this size
            before dispatching to ``forward``.
        output_dir (str | Path | None): Root output directory used by reducer-style
            runners to write shard-local files and merged outputs.
        shard_subdir (str): Optional sub-path under ``output_dir`` or
            the shard-state directory.
        resume (bool): Whether to skip shards that already have a ``done`` marker.

    Notes:
        - When ``output_dir`` is set, shard state is written under
          ``output_dir / "_shards" / shard_subdir / "split.N"``.
        - A shard is considered complete when its directory contains a ``done``
          marker and ``_state.pkl``.
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

    def merge_state(self, states: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Combine in-memory results from finalized shard states."""
        return None

    def merge_shard_files(
        self,
        shard_dirs: List[Path],
        states: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Concatenate per-shard files into final outputs."""
        return None

    @staticmethod
    def _shards_root(output_dir: Path, shard_subdir: str = "") -> Path:
        root = Path(output_dir) / "_shards"
        if shard_subdir:
            root = root / shard_subdir
        return root

    @classmethod
    def _resolve_shard_dir(
        cls,
        output_dir: Optional[str],
        shard_subdir: str,
        shard_id: int,
    ) -> Optional[Path]:
        if output_dir is None:
            return None
        root = cls._shards_root(Path(output_dir), shard_subdir)
        return root / f"split.{shard_id}"

    @classmethod
    def _manifest_path(
        cls,
        output_dir: Optional[str | Path],
        shard_subdir: str = "",
    ) -> Optional[Path]:
        if output_dir is None:
            return None
        return cls._shards_root(Path(output_dir), shard_subdir) / "manifest.json"

    @staticmethod
    def _done_path(shard_dir: Path) -> Path:
        return Path(shard_dir) / "done"

    @classmethod
    def _mark_done(cls, shard_dir: Path) -> None:
        cls._done_path(shard_dir).write_text("", encoding="utf-8")

    @classmethod
    def _clear_done(cls, shard_dir: Path) -> None:
        cls._done_path(shard_dir).unlink(missing_ok=True)

    @classmethod
    def is_shard_done(cls, shard_dir: Path) -> bool:
        return cls._done_path(shard_dir).exists()

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
        shard_dir = cls._resolve_shard_dir(output_dir, shard_subdir, shard_id)
        if shard_dir is not None:
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

        state_result = self.merge_state(states)
        files = self.merge_shard_files(shard_dirs, states) if shard_dirs else None

        if state_result is not None or files is not None:
            merged: Dict[str, Any] = {}
            if isinstance(state_result, dict):
                merged.update(state_result)
            if isinstance(files, dict):
                merged.update(files)
            return merged or (files if files is not None else state_result)

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

    def _manifest_data(self, shards: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "version": 1,
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "shard_subdir": self.shard_subdir,
            "shards": list(shards),
        }

    def _write_manifest(self, shards: Sequence[Dict[str, Any]]) -> Optional[Path]:
        manifest_path = self._manifest_path(self.output_dir, self.shard_subdir)
        if manifest_path is None:
            return None
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self._manifest_data(shards), f, ensure_ascii=False, indent=2)
        return manifest_path

    def _plan_shards(self, items: Sequence[Any]) -> List[Dict[str, Any]]:
        par_config = get_parallel_config()
        env = getattr(par_config, "env", "local") if par_config is not None else "local"
        num_shards = 1
        if par_config is not None and env not in ("local",):
            num_shards = int(getattr(par_config, "n_workers", 1))
        chunks = _default_chunk(list(items), num_shards)
        if not chunks:
            return []
        return [
            {"shard_id": shard_id, "items": list(chunk)}
            for shard_id, chunk in enumerate(chunks)
        ]

    def _pending_shards(self, shards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.output_dir is None or not self.resume:
            return list(shards)

        pending = []
        for shard in shards:
            shard_dir = self._resolve_shard_dir(
                str(self.output_dir), self.shard_subdir, int(shard["shard_id"])
            )
            if shard_dir is None or not self.is_shard_done(shard_dir):
                pending.append(shard)
        return pending

    def _completed_states_from_shards(
        self,
        shards: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        states = []
        for shard in shards:
            shard_dir = self._resolve_shard_dir(
                str(self.output_dir) if self.output_dir is not None else None,
                self.shard_subdir,
                int(shard["shard_id"]),
            )
            if shard_dir is None:
                continue
            state_path = Path(shard_dir) / "_state.pkl"
            if not self.is_shard_done(shard_dir):
                raise FileNotFoundError(f"Shard {shard['shard_id']} is not complete: {shard_dir}")
            if not state_path.exists():
                raise FileNotFoundError(
                    f"Shard {shard['shard_id']} is missing state: {state_path}"
                )
            states.append(self.deserialize_state(shard_dir))
        return states

    def _augment_env(self, env: Dict[str, Any]) -> Dict[str, Any]:
        if self.output_dir is not None:
            env.setdefault("output_dir", str(self.output_dir))
        if self.shard_subdir:
            env.setdefault("shard_subdir", self.shard_subdir)
        return env

    @classmethod
    def _run_one_shard(
        cls,
        shard_id: int,
        items: Sequence[Any],
        env: Dict[str, Any],
    ) -> Dict[str, Any]:
        state = cls.init_state(shard_id=shard_id, **env)
        shard_dir = Path(state["shard_dir"]) if state.get("shard_dir") else None
        if shard_dir is not None:
            cls._clear_done(shard_dir)
        for item in items:
            result = cls.forward(item, **env)
            state = cls.reduce_state(state, result, shard_id=shard_id, **env)
        state = cls.finalize_state(state, shard_id=shard_id, **env)
        if shard_dir is not None:
            cls.serialize_state(state, shard_dir)
            cls._mark_done(shard_dir)
        return state

    def _run_local(self, shards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run pending shards sequentially on the driver.

        Notes:
            - Uses ``tqdm`` progress bar over the pending shard list.
            - Each completed shard writes ``_state.pkl`` and ``done`` when
              ``output_dir`` is configured.
        """
        env = self._augment_env(self.provider.build_env_local())
        cls = self.__class__
        states = []
        for shard in tqdm(shards, total=len(shards), desc="shards"):
            states.append(
                cls._run_one_shard(int(shard["shard_id"]), shard["items"], env)
            )
        return states

    def _run_parallel_dask(self, shards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run pending shards with Dask workers and persist shard-local state."""
        from dask.distributed import as_completed

        par_config = get_parallel_config()
        if not shards:
            return []

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
        """Dispatch execution according to the configured parallel mode.

        Args:
            indices (Iterable[int]): Indices to process.

        Notes:
            - If no parallel config is set or ``env='local'``, run locally.
            - Otherwise, run in parallel with environment injection.
        """
        indices = list(indices)
        if self.batch_size is not None:
            if self.batch_size <= 0:
                raise ValueError("batch_size must be a positive integer.")
            indices = _chunk_indices(indices, self.batch_size)
        shards = self._plan_shards(indices)
        self._write_manifest(shards)
        pending = self._pending_shards(shards)

        par_config = get_parallel_config()
        transient_states: List[Dict[str, Any]] = []
        if pending:
            if par_config is None or getattr(par_config, "env", "local") == "local":
                transient_states = self._run_local(pending)
            else:
                transient_states = self._run_parallel_dask(pending)

        if self.output_dir is not None:
            return self.merge_states(self._completed_states_from_shards(shards))
        return self.merge_states(transient_states)


def _chunk_indices(indices: Sequence[int], batch_size: int) -> List[List[int]]:
    """Split a sequence of indices into fixed-size chunks."""
    batches = []
    for i in range(0, len(indices), batch_size):
        batches.append(list(indices[i : i + batch_size]))
    return [b for b in batches if b]
