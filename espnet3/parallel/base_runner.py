"""BaseRunner class for orchestrating local and parallel shard executions."""

import json
import logging
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
        bool: ``True`` if at least one shard fragment was found and written,
        ``False`` otherwise.

    Raises:
        OSError: If the output parent directory cannot be created.

    Example:
        >>> from pathlib import Path
        >>> shard_dirs = [Path("output/split.0"), Path("output/split.1")]
        >>> found = concatenate_shard_files(
        ...     shard_dirs, "hyp.scp", Path("output/hyp.scp")
        ... )
        >>> print(found)  # True if at least one shard contained hyp.scp
        True
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

    This class is the durable parallel execution path in ESPnet3. It writes
    shard-local files under ``output_dir`` and supports resume by reusing shard
    state when a previous run completed some splits.

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
            runners to write shard-local files and merged outputs. This must be
            provided before calling the runner.
        shard_subdir (str): Optional sub-path under ``output_dir`` where shard
            directories and manifest are written.
        resume (bool): Whether to skip shards that already have a ``done`` marker.

    Notes:
        - When ``output_dir`` is set, shard state is written under
          ``output_dir / shard_subdir / "split.N"``.
        - A shard is considered complete when its directory contains a ``done``
          marker.

    Example:
        >>> from espnet3.parallel.env_provider import EnvironmentProvider
        >>> from omegaconf import OmegaConf
        >>> class MyProvider(EnvironmentProvider):
        ...     def build_env_local(self):
        ...         return {"dataset": ..., "model": ...}
        ...     def build_worker_setup_fn(self):
        ...         def setup():
        ...             return {"dataset": ..., "model": ...}
        ...         return setup
        >>> class MyRunner(BaseRunner):
        ...     @staticmethod
        ...     def forward(idx, dataset, model, **env):
        ...         return model(dataset[idx])
        >>> provider = MyProvider(OmegaConf.create({}))
        >>> runner = MyRunner(provider, output_dir="/tmp/out")
        >>> runner(range(100))
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
        """Open per-shard file writers before processing begins.

        Called once at the start of each shard. Override to open output
        file handles or other resources that accumulate results across
        multiple ``forward`` calls within a shard.

        Args:
            shard_dir: Directory dedicated to this shard's output files.
            **env: Full worker environment (dataset, model, etc.).

        Returns:
            Dict[str, Any]: A writers dict passed to every ``write_record``
            and ``close_writers`` call for this shard.
        """
        return {}

    @staticmethod
    def write_record(
        writers: Dict[str, Any],
        result: Any,
        state: Dict[str, Any],
        **env,
    ) -> None:
        """Persist one ``forward`` result into the shard state or files.

        Called after each ``forward`` invocation. Override to stream results
        into open file handles instead of accumulating them in memory.

        Args:
            writers: The dict returned by ``open_writers`` for this shard.
            result: The value returned by ``forward`` for this item.
            state: Mutable shard state dict (``records``, ``shard_id``, etc.).
            **env: Full worker environment.
        """
        state.setdefault("records", []).append(result)

    @staticmethod
    def close_writers(writers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Close per-shard writers after all items are processed.

        Called once at the end of each shard. Override to flush and close
        any file handles opened in ``open_writers``. The optional return
        value is merged into the shard state before it is persisted.

        Args:
            writers: The dict returned by ``open_writers`` for this shard.

        Returns:
            Optional[Dict[str, Any]]: Extra entries to merge into the shard
            state, or ``None``.
        """
        for writer in writers.values():
            close = getattr(writer, "close", None)
            if callable(close):
                close()
        return None

    def merge(self, shard_dirs: List[Path]) -> Any:
        """Merge completed shard outputs into the final result.

        Called on the driver after all shards finish. Override to aggregate
        per-shard files (e.g., SCP files, stats arrays) into a single output.

        Args:
            shard_dirs: Ordered list of completed shard directories.

        Returns:
            Any: Aggregated result, or ``None`` if outputs are written to
            disk and no in-memory result is needed.
        """
        return None

    @staticmethod
    def _get_shards_root(output_dir: Path, shard_subdir: str = "") -> Path:
        """Return the root directory for all shard subdirectories under output_dir."""
        root = Path(output_dir)
        if shard_subdir:
            root = root / shard_subdir
        return root

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
    def is_shard_done(cls, shard_dir: Path) -> bool:
        """Return True if the shard has a completion marker file."""
        return cls._get_done_path(shard_dir).exists()

    @classmethod
    def init_state(
        cls,
        shard_id: int = 0,
        output_dir: str = "",
        shard_subdir: str = "",
        **env,
    ) -> Dict[str, Any]:
        """Build the initial state dict for one shard and open its writers.

        Args:
            shard_id: Zero-based shard index.
            output_dir: Root output directory (string form).
            shard_subdir: Optional sub-path appended to ``output_dir``.
            **env: Full worker environment forwarded to ``open_writers``.

        Returns:
            Dict[str, Any]: Initial state with keys ``shard_id``,
            ``shard_dir``, ``_writers``, and ``records``.

        Example:
            >>> state = MyRunner.init_state(
            ...     shard_id=0, output_dir="/tmp/out", dataset=ds, model=md
            ... )
            >>> print(state["shard_dir"])
            /tmp/out/split.0
        """
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
        """Fold a single ``forward`` result into the shard state.

        Args:
            state: Current shard state (mutated in place via ``write_record``).
            result: Value returned by ``forward`` for one item or batch.
            **env: Full worker environment.

        Returns:
            Dict[str, Any]: Updated shard state.
        """
        cls.write_record(state["_writers"], result, state, **env)
        return state

    @classmethod
    def finalize_state(cls, state: Dict[str, Any], **env) -> Dict[str, Any]:
        """Close writers and finalize the shard state.

        Args:
            state: Shard state containing ``_writers`` and accumulated data.
            **env: Full worker environment.

        Returns:
            Dict[str, Any]: Finalized state with ``_writers`` removed and any
            extra metadata from ``close_writers`` merged in.
        """
        meta = cls.close_writers(state.get("_writers", {})) or {}
        if meta:
            state.update(meta)
        state.pop("_writers", None)
        return state

    def _write_manifest(self, shards: Sequence[Dict[str, Any]]) -> Path:
        """Write shard plan to manifest.json before dispatch in __call__."""
        manifest_path = (
            self._get_shards_root(self.output_dir, self.shard_subdir) / "manifest.json"
        )
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

    def _filter_pending_shards(
        self, shards: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return pending shards; skips done-marked ones when resume=True."""
        if not self.resume:
            return list(shards)

        pending = []
        for shard in shards:
            shard_dir = self._resolve_shard_dir(
                str(self.output_dir), self.shard_subdir, int(shard["shard_id"])
            )
            if not self.is_shard_done(shard_dir):
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
        cls._get_done_path(shard_dir).unlink(missing_ok=True)
        for item in items:
            result = cls.forward(item, **env)
            state = cls.reduce_state(state, result, shard_id=shard_id, **env)
        cls.finalize_state(state, shard_id=shard_id, **env)
        cls._get_done_path(shard_dir).write_text("", encoding="utf-8")
        return state

    def _run_local(self, shards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run pending shards sequentially on the driver.

        Notes:
            - Uses ``tqdm`` progress bar over the pending shard list.
            - Each completed shard writes its shard files and ``done``.
        """
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
        """Dispatch execution according to the configured parallel mode.

        Splits ``indices`` into shards, runs pending shards locally or via
        Dask, then calls :meth:`merge` on all completed shard directories.

        Args:
            indices (Iterable[int]): Indices (or pre-batched index lists) to
                process. Pass batched lists when ``batch_size`` is ``None``
                and batching is handled externally.

        Returns:
            Any: The value returned by :meth:`merge`, which is
            ``None`` by default and a dict or list in reducer subclasses.

        Raises:
            RuntimeError: If ``output_dir`` was not set on construction.
            ValueError: If ``batch_size`` is not a positive integer.
            FileNotFoundError: If a shard directory is missing its ``done``
                marker after execution (indicates a failed shard).

        Notes:
            - If no parallel config is set or ``env='local'``, run locally.
            - Otherwise, run in parallel with environment injection.

        Example:
            >>> runner = MyRunner(provider, output_dir="/tmp/out")
            >>> result = runner(range(200))
        """
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
        shards = self._plan_shards(indices)
        self._write_manifest(shards)
        pending = self._filter_pending_shards(shards)

        par_config = get_parallel_config()
        if pending:
            if par_config is None or getattr(par_config, "env", "local") == "local":
                self._run_local(pending)
            else:
                self._run_parallel_dask(pending)

        return self.merge(self._get_completed_shard_dirs(shards))
