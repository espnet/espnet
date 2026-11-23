"""Datasets selected from omniio dumps with SQL and decoded on access.

An omniio dump keeps each modality of a corpus in its own omniio ``Blob``
(``blob_<n>.bin`` files plus one ``metadata.parquet``), laid out as::

    <dump_root>/<modality>/<corpus>[/<subset>]/metadata.parquet

for example ``<dump_root>/audio/librispeech/train-clean-100`` and
``<dump_root>/text/librispeech/train-clean-100``. Items that belong together
share an ``id`` across modalities, and omniio's metadata already records each
item's size (``samples`` for audio), so no separate pass is needed to find
item lengths.

:class:`OmniIODataset` takes a DuckDB query over those ``metadata.parquet``
files, so a recipe config selects, joins and filters its data on the fly.
Relative paths resolve against the dump root, and ``$name`` parameters come
from ``params``::

    SELECT a.id,
           a.path AS speech_path, a.start_byte AS speech_start,
           a.end_byte - a.start_byte AS speech_size, a.samples AS length,
           t.path AS text_path, t.start_byte AS text_start,
           t.end_byte - t.start_byte AS text_size
    FROM read_parquet('audio/librispeech/' || $split || '/metadata.parquet') a
    JOIN read_parquet('text/librispeech/' || $split || '/metadata.parquet') t
      USING (id)
    WHERE a.samples <= 20 * 16000
    ORDER BY id

Each row is one item. For each declared field ``<f>`` the query returns
either ``<f>_path``, ``<f>_start`` and ``<f>_size``, a byte range decoded with
``omniio.interface``, or a column named ``<f>`` whose value is used as-is
(e.g. a transcript kept inline in a parquet table). An optional ``length``
column gives the per-item length used for batching.

With ``total_shards > 1`` the rows are split by ``hash(<shard_by>)`` and
:meth:`ShardedOmniIODataset.shard` runs the query for one shard only, so each
GPU holds the index of its own shard and never the full one.
"""

from __future__ import annotations

import copy
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from espnet3.components.data.base_dataset import BaseDataset
from espnet3.components.data.dataset import ShardedDataset
from espnet3.contract import Field

logger = logging.getLogger(__name__)

DUMP_ROOT_ENV = "OMNIIO_DUMP_ROOT"
METADATA_FILE = "metadata.parquet"
BLOB_SUFFIXES = ("_path", "_start", "_size")


def _read_audio(path: str, start: int, size: int) -> np.ndarray:
    """Decode an omniio audio entry; mono audio comes back one-dimensional."""
    from omniio.interface import audio_read

    array = audio_read(path, start, size).array
    return array[:, 0] if array.shape[1] == 1 else array


def _read_text(path: str, start: int, size: int) -> str:
    """Decode an omniio text entry."""
    from omniio.interface import text_read

    return text_read(path, start, size).text


# omniio modality -> reader. Each name is also the ESPnet3 field kind of the
# decoded value. omniio can also read image, video and MIDI entries, but
# ESPnet3 has no field kinds for them yet; add a reader here together with the
# kind in espnet3.contract when a task needs one.
_READERS: dict[str, Callable[[str, int, int], Any]] = {
    "audio": _read_audio,
    "text": _read_text,
}


def resolve_dump_root(dump_root: str | Path | None = None) -> Path:
    """Return the omniio dump root: ``dump_root``, else ``$OMNIIO_DUMP_ROOT``.

    Raises:
        ValueError: If neither ``dump_root`` nor ``$OMNIIO_DUMP_ROOT`` is set.

    Examples:
        >>> resolve_dump_root("/data/omniio_dump")
        PosixPath('/data/omniio_dump')
    """
    if dump_root is None:
        dump_root = os.environ.get(DUMP_ROOT_ENV) or None
    if dump_root is None:
        raise ValueError(
            f"No omniio dump root: pass dump_root or set ${DUMP_ROOT_ENV}."
        )
    return Path(dump_root)


def get_blob_dir(
    dump_root: str | Path | None,
    modality: str,
    corpus: str,
    subset: str | None = None,
) -> Path:
    """Return ``<dump_root>/<modality>/<corpus>[/<subset>]``.

    Examples:
        >>> get_blob_dir("/d", "audio", "librispeech", "dev-clean")
        PosixPath('/d/audio/librispeech/dev-clean')
    """
    path = resolve_dump_root(dump_root) / modality / corpus
    return path / subset if subset else path


def is_blob_dumped(path: str | Path) -> bool:
    """Return whether ``path`` holds a finished blob.

    ``Blob.append`` writes ``metadata.parquet`` only after every worker shard
    has been merged, so its presence marks a completed append.

    Examples:
        >>> is_blob_dumped("/nonexistent/audio/librispeech/dev-clean")
        False
    """
    return (Path(path) / METADATA_FILE).is_file()


def dump_blob(
    path: str | Path,
    modality: str,
    items: Sequence[Any],
    ids: Sequence[str],
    num_workers: int = 0,
    **modality_kwargs,
) -> None:
    """Append ``items`` to the blob at ``path`` through ``Blob.append``.

    IDs already in the blob are skipped, so an interrupted dump can be re-run
    and only writes what is missing. Any item omniio fails to encode aborts
    the dump: a silently incomplete blob would break joins on ``id``.

    Args:
        path: Blob directory, normally from :func:`get_blob_dir`.
        modality: omniio modality (``"audio"``, ``"text"``, ...).
        items: Items accepted by that modality's omniio writer, e.g. audio
            file paths, or strings with ``is_path=False`` for text.
        ids: One unique ID per item, shared with the corpus's other
            modalities.
        num_workers: ``Blob.append`` worker processes; 0 runs in-process.
        **modality_kwargs: Forwarded to the omniio writer, e.g.
            ``target_format`` for audio or ``compression_level`` for text.

    Raises:
        ValueError: If ``items`` and ``ids`` differ in length.

    Examples:
        >>> dump_blob(
        ...     get_blob_dir("/data/omniio_dump", "text", "librispeech", "dev-clean"),
        ...     "text",
        ...     ["MISTER QUILTER IS THE APOSTLE"],
        ...     ["1272-128104-0000"],
        ...     is_path=False,
        ... )  # doctest: +SKIP
    """
    from omniio.blob.blob import Blob

    if len(items) != len(ids):
        raise ValueError(f"{len(items)} items but {len(ids)} ids for {path}")
    blob = Blob(archive_dir=str(path), modality=modality)
    # Drop IDs already in the blob here: Blob.append would skip them too, but
    # with one warning per ID.
    existing = set(blob.get_metadata().column("id").to_pylist()) if len(blob) else set()
    todo = [(item, uid) for item, uid in zip(items, ids) if uid not in existing]
    logger.info(
        "Dumping %d %s items to %s (%d already present)",
        len(todo),
        modality,
        path,
        len(items) - len(todo),
    )
    if not todo:
        return
    blob.append(
        items=[item for item, _ in todo],
        ids=[uid for _, uid in todo],
        num_workers=num_workers,
        **modality_kwargs,
    )


def _quote_identifier(name: str) -> str:
    """Return ``name`` as a double-quoted SQL identifier.

    DuckDB binds values as parameters but has no parameter form for column
    names, so the ``shard_by`` column is quoted here.

    Examples:
        >>> _quote_identifier('a"b')
        '"a""b"'
    """
    return '"' + name.replace('"', '""') + '"'


def _get_query_parameters(query: str) -> set[str]:
    """Return the ``$name`` parameters ``query`` uses.

    Raises:
        ValueError: If ``query`` is not exactly one SQL statement.

    Examples:
        >>> sorted(_get_query_parameters("SELECT $b AS x, $a AS y;"))
        ['a', 'b']
    """
    import duckdb

    statements = duckdb.extract_statements(query)
    if len(statements) != 1:
        raise ValueError(
            f"The query must be one SQL statement, got {len(statements)}: {query}"
        )
    return set(statements[0].named_parameters)


def _fetch_table(relation):
    """Run a DuckDB relation and return its rows as a pyarrow Table."""
    # DuckDB 2.0 has to_arrow_table(); 1.x relations have arrow().
    fetch = getattr(relation, "to_arrow_table", None) or relation.arrow
    return fetch()


class OmniIODataset(BaseDataset):
    """Items selected by a SQL query over omniio dumps, decoded on access.

    Args:
        query: One DuckDB statement returning one row per item: an ``id``
            column, and for each field either ``<f>_path``/``<f>_start``/
            ``<f>_size`` (read from an omniio blob) or an inline ``<f>``
            column. Relative file paths resolve against the dump root, and
            ``$name`` parameters are bound from ``params``.
        fields: Field name -> omniio modality (``audio`` or ``text``). The
            modality is also the field's ESPnet3 kind.
        params: Values for the query's ``$name`` parameters. Entries the
            query does not use are ignored; ``$dump_root`` is filled from
            ``dump_root`` when not given.
        dump_root: omniio dump root that relative paths resolve against;
            defaults to ``$OMNIIO_DUMP_ROOT``. Without either, relative paths
            resolve against the working directory.
        total_shards: Number of shards the rows are split into, normally the
            distributed world size. ``None`` or 1 keeps one unsharded index.
        dist_world_size: Distributed world size the shards rotate over; see
            ``DataLoaderBuilder``. Defaults to ``total_shards``.
        shard_by: Column whose hash assigns a row to a shard. Rows with the
            same value stay in the same shard (e.g. one speaker).
        length_column: Column with each item's length, used for batching
            by :meth:`get_item_lengths`; it may be absent.
        transforms: Optional field name -> callable applied to the decoded
            value.

    Notes:
        The full index is built only when it is needed (``__getitem__`` past
        the first item, :meth:`get_item_lengths`, ...). A sharded dataset is
        normally only asked for its length and first item before
        ``DataLoaderBuilder`` calls :meth:`shard`, so a training rank holds
        just its own shard.

    Raises:
        ValueError: If ``fields`` is empty or names a modality that cannot be
            decoded, the query is not one statement, a ``$name`` parameter
            has no value, or the query lacks a required column.

    Examples:
        Select one split of a dump and read an item:

        >>> ds = OmniIODataset(
        ...     query=(
        ...         "SELECT id, path AS text_path, start_byte AS text_start, "
        ...         "end_byte - start_byte AS text_size FROM read_parquet("
        ...         "'text/librispeech/' || $split || '/metadata.parquet') "
        ...         "ORDER BY id"
        ...     ),
        ...     fields={"text": "text"},
        ...     params={"split": "dev-clean"},
        ...     dump_root="/data/omniio_dump",
        ... )  # doctest: +SKIP
        >>> ds.get_by_id("1272-128104-0000")["text"][:6]  # doctest: +SKIP
        'MISTER'

        Inline columns need no dump at all:

        >>> ds = OmniIODataset(
        ...     query="SELECT * FROM (VALUES ('a', 'hi', 2)) t(id, text, length)",
        ...     fields={"text": "text"},
        ... )
        >>> len(ds), ds[0]
        (1, {'text': 'hi'})
    """

    def __new__(cls, *args, total_shards: int | None = None, **kwargs):
        """Return a :class:`ShardedOmniIODataset` when ``total_shards > 1``."""
        if total_shards is not None and int(total_shards) > 1:
            if not issubclass(cls, ShardedOmniIODataset):
                cls = _get_sharded_variant(cls)
        return super().__new__(cls)

    def __init__(
        self,
        query: str,
        fields: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
        dump_root: str | Path | None = None,
        total_shards: int | None = None,
        dist_world_size: int | None = None,
        shard_by: str = "id",
        length_column: str = "length",
        transforms: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> None:
        """Validate the query's columns; the rows are fetched lazily."""
        if not fields:
            raise ValueError("OmniIODataset needs at least one field.")
        self.modalities = dict(fields)
        for name, modality in self.modalities.items():
            if modality not in _READERS:
                raise ValueError(
                    f"Field '{name}': cannot decode modality '{modality}'; "
                    f"supported: {sorted(_READERS)}"
                )
        self.fields = tuple(
            Field(name, modality) for name, modality in self.modalities.items()
        )
        self.query = query
        root = dump_root or os.environ.get(DUMP_ROOT_ENV) or None
        self.dump_root = Path(root) if root is not None else None
        self.params = self._bind_parameters(dict(params or {}))
        # None (not 1) when unsharded: DataLoaderBuilder shards any dataset
        # whose total_shards is set.
        sharded = total_shards is not None and int(total_shards) > 1
        self.total_shards = int(total_shards) if sharded else None
        self.dist_world_size = int(dist_world_size or total_shards) if sharded else None
        self.shard_by = shard_by
        self.length_column = length_column
        self.transforms = dict(transforms or {})
        self.shard_idx: int | None = None

        self._index: dict[str, Any] | None = None
        self._len: int | None = None
        self._head: dict[str, Any] | None = None
        self._id_to_index: dict[str, int] | None = None
        self._last_shard: tuple[int, OmniIODataset] | None = None
        with self._select_rows() as rows:
            self._layout = self._check_columns(rows.columns)

    # ------------------------------------------------------------------ #
    # Index
    # ------------------------------------------------------------------ #

    def _bind_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return the values of the query's ``$name`` parameters.

        Raises:
            ValueError: If a parameter the query uses has no value.
        """
        names = _get_query_parameters(self.query)
        if "dump_root" in names and "dump_root" not in params:
            params["dump_root"] = str(resolve_dump_root(self.dump_root))
        missing = sorted(names - set(params))
        if missing:
            raise ValueError(
                f"The query uses parameters {missing} but params only has "
                f"{sorted(params)}."
            )
        return {name: params[name] for name in names}

    @contextmanager
    def _select_rows(self) -> Iterator[Any]:
        """Yield a DuckDB relation over this dataset's rows: all, or a shard's.

        Each call uses its own connection, so dataloader worker processes
        never share one.
        """
        import duckdb

        con = duckdb.connect()
        try:
            if self.dump_root is not None:
                con.execute("SET file_search_path = ?", [str(self.dump_root)])
            rows = con.sql(self.query, params=self.params)
            if self.shard_idx is not None:
                rows = rows.filter(
                    f"hash({_quote_identifier(self.shard_by)}) "
                    f"% {self.total_shards} = {self.shard_idx}"
                )
            yield rows
        finally:
            con.close()

    def _check_columns(self, columns: Sequence[str]) -> dict[str, str]:
        """Map each field to ``blob`` or ``inline`` from the query's columns."""
        columns = set(columns)
        if "id" not in columns:
            raise ValueError(f"The query must return an 'id' column: {self.query}")
        if self.shard_by not in columns:
            raise ValueError(f"shard_by column '{self.shard_by}' is not returned")
        layout = {}
        for name in self.modalities:
            blob_cols = {name + suffix for suffix in BLOB_SUFFIXES}
            if blob_cols <= columns:
                layout[name] = "blob"
            elif name in columns:
                layout[name] = "inline"
            else:
                raise ValueError(
                    f"Field '{name}': the query returns neither "
                    f"{sorted(blob_cols)} nor an inline '{name}' column."
                )
        return layout

    def _build_index(self, table) -> dict[str, Any]:
        """Turn the fetched rows into compact NumPy columns."""
        import pyarrow as pa

        index: dict[str, Any] = {
            "id": np.asarray(table.column("id").to_pylist(), dtype=str)
        }
        for name, kind in self._layout.items():
            if kind == "inline":
                index[name] = np.asarray(table.column(name).to_pylist(), dtype=object)
                continue
            paths = table.column(f"{name}_path").combine_chunks()
            if not pa.types.is_dictionary(paths.type):
                paths = paths.dictionary_encode()
            index[f"{name}_bins"] = np.asarray(paths.dictionary.to_pylist())
            index[f"{name}_code"] = paths.indices.to_numpy().astype(np.int32)
            for part in ("start", "size"):
                col = table.column(f"{name}_{part}").to_numpy()
                index[f"{name}_{part}"] = col.astype(np.int64)
        if self.length_column in table.column_names:
            index["length"] = table.column(self.length_column).to_numpy()
        return index

    def _ensure_index(self) -> dict[str, Any]:
        """Fetch the rows into the in-memory index once; return the index."""
        if self._index is None:
            with self._select_rows() as rows:
                table = _fetch_table(rows)
            self._index = self._build_index(table)
            self._len = len(self._index["id"])
            logger.info(
                "OmniIODataset: %d rows%s",
                self._len,
                (
                    ""
                    if self.shard_idx is None
                    else f" in shard {self.shard_idx}/{self.total_shards}"
                ),
            )
        return self._index

    def __len__(self) -> int:
        """Return the number of rows (a COUNT query until the index is built)."""
        if self._len is None:
            with self._select_rows() as rows:
                self._len = int(rows.aggregate("count(*)").fetchall()[0][0])
        return self._len

    def keys(self) -> list[str]:
        """Return the item IDs in row order.

        ``CombinedDataset`` uses this to look items up by ID.

        Examples:
            >>> OmniIODataset(
            ...     query="SELECT * FROM (VALUES ('b', 'x'), ('a', 'y')) t(id, text)",
            ...     fields={"text": "text"},
            ... ).keys()
            ['b', 'a']
        """
        return self._ensure_index()["id"].tolist()

    def get_item_lengths(self) -> np.ndarray:
        """Return every item's ``length`` column value, in row order.

        ``DataLoaderBuilder`` batches by these when
        ``batches.lengths_from_dataset`` is set.

        Raises:
            KeyError: If the query has no length column.

        Examples:
            >>> OmniIODataset(
            ...     query=(
            ...         "SELECT * FROM (VALUES ('a', 'x', 3), ('b', 'y', 5)) "
            ...         "t(id, text, length)"
            ...     ),
            ...     fields={"text": "text"},
            ... ).get_item_lengths().tolist()
            [3, 5]
        """
        index = self._ensure_index()
        if "length" not in index:
            raise KeyError(
                f"The query returns no '{self.length_column}' column to batch by."
            )
        return index["length"]

    # ------------------------------------------------------------------ #
    # Items
    # ------------------------------------------------------------------ #

    def _decode(self, name: str, row: Mapping[str, Any]) -> Any:
        """Decode field ``name`` from its inline value or blob byte range."""
        if self._layout[name] == "inline":
            value = row[name]
        else:
            read = _READERS[self.modalities[name]]
            value = read(row["path"], int(row["start"]), int(row["size"]))
        transform = self.transforms.get(name)
        return transform(value) if transform is not None else value

    def read_field(self, name: str, idx: int) -> Any:
        """Decode field ``name`` of the item at ``idx``.

        Examples:
            >>> OmniIODataset(
            ...     query="SELECT * FROM (VALUES ('a', 'hi')) t(id, text)",
            ...     fields={"text": "text"},
            ... ).read_field("text", 0)
            'hi'
        """
        index = self._ensure_index()
        idx = int(idx)
        if self._layout[name] == "inline":
            row = {name: index[name][idx]}
        else:
            row = {
                "path": str(index[f"{name}_bins"][index[f"{name}_code"][idx]]),
                "start": index[f"{name}_start"][idx],
                "size": index[f"{name}_size"][idx],
            }
        return self._decode(name, row)

    def _read_first_item(self) -> dict[str, Any]:
        """Return item 0 from a ``LIMIT 1`` query, without building the index.

        Only the raw row is cached; it is decoded (and transformed) on every
        call, like any other item.
        """
        if self._head is None:
            with self._select_rows() as rows:
                row = _fetch_table(rows.limit(1))
            if row.num_rows == 0:
                raise IndexError("The query returned no rows.")
            row = row.to_pylist()[0]
            self._head = {
                name: (
                    {name: row[name]}
                    if kind == "inline"
                    else {
                        "path": row[f"{name}_path"],
                        "start": row[f"{name}_start"],
                        "size": row[f"{name}_size"],
                    }
                )
                for name, kind in self._layout.items()
            }
        return {name: self._decode(name, raw) for name, raw in self._head.items()}

    def get_by_id(self, item_id: str) -> dict[str, Any]:
        """Return the item whose ID is ``item_id``.

        Raises:
            ValueError: If the query returns the same ID more than once.
            KeyError: If no row has ID ``item_id``.

        Examples:
            >>> OmniIODataset(
            ...     query="SELECT * FROM (VALUES ('a', 'x'), ('b', 'y')) t(id, text)",
            ...     fields={"text": "text"},
            ... ).get_by_id("b")
            {'text': 'y'}
        """
        if self._id_to_index is None:
            keys = self.keys()
            id_to_index = {uid: i for i, uid in enumerate(keys)}
            if len(id_to_index) != len(keys):
                ids, counts = np.unique(keys, return_counts=True)
                raise ValueError(
                    "The query returns duplicate IDs, so items cannot be "
                    f"looked up by ID: {ids[counts > 1][:5].tolist()}"
                )
            self._id_to_index = id_to_index
        return self[self._id_to_index[item_id]]

    def __getitem__(self, idx: int | str) -> dict[str, Any]:
        """Return the item at integer index ``idx``, or with ID ``idx``."""
        if isinstance(idx, str):
            return self.get_by_id(idx)
        if self._index is None and int(idx) == 0:
            # CombinedDataset probes item 0 of every dataset; answer it
            # without fetching every row of an unsharded training query.
            return self._read_first_item()
        if not 0 <= int(idx) < len(self):
            raise IndexError(f"index {idx} out of range for {len(self)} items")
        return {name: self.read_field(name, idx) for name in self.modalities}

    def iter_field(self, name: str) -> Iterable[Any]:
        """Yield one field for every item, e.g. all transcripts.

        Examples:
            >>> list(OmniIODataset(
            ...     query="SELECT * FROM (VALUES ('a', 'x'), ('b', 'y')) t(id, text)",
            ...     fields={"text": "text"},
            ... ).iter_field("text"))
            ['x', 'y']
        """
        for idx in range(len(self)):
            yield self.read_field(name, idx)


class ShardedOmniIODataset(OmniIODataset):
    """An :class:`OmniIODataset` split into ``total_shards`` by row hash.

    Created by ``OmniIODataset(..., total_shards=N)`` with ``N > 1``.
    :meth:`shard` returns a dataset whose index holds only that shard's rows.

    Examples:
        >>> ds = OmniIODataset(
        ...     query="SELECT range::VARCHAR AS id, 'x' AS text FROM range(10)",
        ...     fields={"text": "text"},
        ...     total_shards=2,
        ... )
        >>> isinstance(ds, ShardedOmniIODataset)
        True
        >>> len(ds.shard(0)) + len(ds.shard(1)) == len(ds) == 10
        True
    """

    def shard(self, idx: int) -> OmniIODataset:
        """Return shard ``idx``: the rows with ``hash(shard_by) % N == idx``.

        Raises:
            IndexError: If ``idx`` is not in ``[0, total_shards)``.
        """
        idx = int(idx)
        if not 0 <= idx < self.total_shards:
            raise IndexError(f"shard {idx} out of range for {self.total_shards}")
        # The same shard is requested again every epoch when
        # total_shards == world size; keep the last one instead of re-querying.
        if self._last_shard is not None and self._last_shard[0] == idx:
            return self._last_shard[1]
        shard = copy.copy(self)
        shard.shard_idx = idx
        shard._index = shard._len = shard._head = None
        shard._id_to_index = shard._last_shard = None
        shard._ensure_index()
        self._last_shard = (idx, shard)
        return shard


ShardedDataset.register(ShardedOmniIODataset)
_SHARDED_VARIANTS: dict[type, type] = {OmniIODataset: ShardedOmniIODataset}


def _get_sharded_variant(cls: type) -> type:
    """Return a subclass of ``cls`` that is also a :class:`ShardedOmniIODataset`."""
    if cls not in _SHARDED_VARIANTS:
        variant = type(f"Sharded{cls.__name__}", (cls, ShardedOmniIODataset), {})
        variant.__module__ = cls.__module__
        ShardedDataset.register(variant)
        _SHARDED_VARIANTS[cls] = variant
    return _SHARDED_VARIANTS[cls]
