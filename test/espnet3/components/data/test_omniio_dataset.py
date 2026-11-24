"""Tests for espnet3.components.data.omniio_dataset."""

import pickle

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("omniio")
pytest.importorskip("duckdb")

from espnet3.components.data.dataset import ShardedDataset  # noqa: E402
from espnet3.components.data.omniio_dataset import (  # noqa: E402
    DUMP_COMPLETE_FILE,
    DUMP_ROOT_ENV,
    OmniIODataset,
    ShardedOmniIODataset,
    dump_blob,
    get_blob_dir,
    is_blob_dumped,
    resolve_dump_root,
)

SR = 16000
N_AUDIO = 12

PAIRED = """
SELECT a.id,
       a.path AS speech_path, a.start_byte AS speech_start,
       a.end_byte - a.start_byte AS speech_size, a.samples AS length,
       t.path AS text_path, t.start_byte AS text_start,
       t.end_byte - t.start_byte AS text_size
FROM read_parquet('audio/toy/' || $split || '/metadata.parquet') AS a
JOIN read_parquet('text/toy/' || $split || '/metadata.parquet') AS t USING (id)
{where}
ORDER BY a.id
"""


@pytest.fixture
def dump(tmp_path):
    """Build a dump: audio for u00-u11, text for every even uid plus an orphan."""
    root = tmp_path / "dump"
    rng = np.random.default_rng(0)
    audio = {}
    for i in range(N_AUDIO):
        uid = f"u{i:02d}"
        wave = rng.uniform(-0.5, 0.5, SR // 8 * (i + 1)).astype(np.float32)
        path = tmp_path / f"{uid}.flac"
        sf.write(path, wave, SR, subtype="PCM_16")
        audio[uid] = sf.read(path, dtype="float32")[0]
    dump_blob(
        get_blob_dir(root, "audio", "toy", "train"),
        "audio",
        [str(tmp_path / f"{uid}.flac") for uid in audio],
        list(audio),
        num_workers=2,
    )
    texts = {uid: f"TEXT {uid}" for uid in list(audio)[::2]}
    texts["orphan"] = "NO AUDIO"
    dump_blob(
        get_blob_dir(root, "text", "toy", "train"),
        "text",
        list(texts.values()),
        list(texts),
        is_path=False,
        compression_level=3,
    )
    return root, audio, texts


def _paired(root, where="", **kwargs):
    """Return the audio/text dataset joined on ID, optionally filtered."""
    return OmniIODataset(
        query=PAIRED.format(where=where),
        fields={"speech": "audio", "text": "text"},
        params={"split": "train"},
        dump_root=root,
        **kwargs,
    )


def test_resolve_dump_root(monkeypatch, tmp_path):
    """The dump root comes from the argument, else $OMNIIO_DUMP_ROOT."""
    monkeypatch.delenv(DUMP_ROOT_ENV, raising=False)
    with pytest.raises(ValueError):
        resolve_dump_root()
    monkeypatch.setenv(DUMP_ROOT_ENV, str(tmp_path))
    assert resolve_dump_root() == tmp_path


def test_blob_layout(dump):
    """Blob dirs follow <root>/<modality>/<corpus>/<split>."""
    root, _, _ = dump
    assert (
        get_blob_dir(root, "audio", "toy", "train") == root / "audio" / "toy" / "train"
    )
    assert is_blob_dumped(root / "audio" / "toy" / "train")
    assert not is_blob_dumped(root / "audio" / "toy" / "missing")


def test_query_parameters(dump):
    """$name parameters are bound from params; unused params are ignored."""
    root, _, _ = dump
    query = (
        "SELECT id, id AS text FROM "
        "read_parquet('audio/toy/' || $split || '/metadata.parquet') "
        "WHERE id <= $last ORDER BY id"
    )
    ds = OmniIODataset(
        query=query,
        fields={"text": "text"},
        params={"split": "train", "last": "u02", "unused": 1},
        dump_root=root,
    )
    assert ds.keys() == ["u00", "u01", "u02"]
    with pytest.raises(ValueError, match=r"parameters \['last'\]"):
        OmniIODataset(
            query=query,
            fields={"text": "text"},
            params={"split": "train"},
            dump_root=root,
        )


def test_query_must_be_one_statement():
    """Several statements are rejected with a clear error."""
    with pytest.raises(ValueError, match="one SQL statement"):
        OmniIODataset(query="SELECT 'a' AS id; SELECT 1", fields={"text": "text"})


def test_join_returns_paired_items(dump):
    """A join on id yields matching audio/text pairs and lengths."""
    root, audio, texts = dump
    ds = _paired(root)
    paired = sorted(set(audio) & set(texts))
    assert len(ds) == len(paired)
    assert ds.keys() == paired
    for idx, uid in enumerate(paired):
        item = ds[idx]
        assert item["text"] == texts[uid]
        assert item["speech"].dtype == np.float32 and item["speech"].ndim == 1
        np.testing.assert_array_equal(item["speech"], audio[uid])
    np.testing.assert_array_equal(
        ds.get_item_lengths(), [len(audio[uid]) for uid in paired]
    )


def test_filter_in_query(dump):
    """A WHERE clause in the query filters rows."""
    root, audio, _ = dump
    ds = _paired(root, where=f"WHERE a.samples > {len(audio['u05'])}")
    assert ds.keys() == ["u06", "u08", "u10"]


def test_inline_column(dump):
    """Inline columns are used as-is."""
    root, _, _ = dump
    ds = OmniIODataset(
        query=(
            "SELECT id, id AS text, 1 AS length FROM "
            "read_parquet('audio/toy/train/metadata.parquet') "
            "ORDER BY id"
        ),
        fields={"text": "text"},
        dump_root=root,
    )
    assert ds[3]["text"] == "u03"
    assert list(ds.iter_field("text"))[:2] == ["u00", "u01"]


def test_lazy_index(dump):
    """len() and item 0 do not build the full index."""
    root, _, texts = dump
    ds = _paired(root)
    assert len(ds) == 6 and ds._index is None  # COUNT query only
    assert ds[0]["text"] == "TEXT u00" and ds._index is None  # LIMIT 1 probe
    ds[1]
    assert ds._index is not None


def test_unsharded_has_no_shard_metadata(dump):
    """total_shards None or 1 gives a plain, unsharded dataset."""
    root, _, _ = dump
    for total in (None, 1):
        ds = _paired(root, total_shards=total)
        assert type(ds) is OmniIODataset
        assert not isinstance(ds, ShardedDataset)
        assert ds.total_shards is None and ds.dist_world_size is None


def test_shards_partition_rows(dump):
    """Shards are disjoint, cover every row and hold only their rows."""
    root, _, _ = dump
    ds = _paired(root, total_shards=3)
    assert isinstance(ds, ShardedOmniIODataset) and isinstance(ds, ShardedDataset)
    assert ds.total_shards == 3 and ds.dist_world_size == 3
    shards = [ds.shard(i) for i in range(3)]
    keys = [s.keys() for s in shards]
    assert sorted(k for ks in keys for k in ks) == ds.keys()
    assert all(not set(a) & set(b) for a in keys for b in keys if a is not b)
    for shard in shards:  # each shard's index holds only its own rows
        assert len(shard._index["id"]) == len(shard)
        for i, uid in enumerate(shard.keys()):
            assert shard[i]["text"] == f"TEXT {uid}"
    assert ds.shard(2) is shards[2]  # the last shard is reused
    with pytest.raises(IndexError):
        ds.shard(3)


def test_shard_by_other_column(dump):
    """Rows go to shard hash(shard_by) % total_shards."""
    root, _, _ = dump
    query = PAIRED.format(where="").replace(
        "SELECT a.id,", "SELECT a.id, CAST(substr(a.id, 2) AS INT) % 3 AS grp,"
    )
    ds = OmniIODataset(
        query=query,
        fields={"speech": "audio", "text": "text"},
        params={"split": "train"},
        dump_root=root,
        total_shards=2,
        shard_by="grp",
    )
    import duckdb

    con = duckdb.connect()
    con.execute("SET file_search_path = ?", [str(root)])

    def run(sql):
        return con.sql(sql, params={"split": "train"}).fetchall()

    full = run(query)
    grp = dict(run(f"SELECT id, grp FROM ({query})"))
    expected = dict(run(f"SELECT id, hash(grp) % 2 FROM ({query})"))
    assert len(set(grp.values())) == 3  # several group values occur
    shard_of = {}
    for i in range(2):
        for uid in ds.shard(i).keys():
            assert uid not in shard_of  # each ID in exactly one shard
            shard_of[uid] = i
    assert len(shard_of) == len(full)
    assert shard_of == expected


def test_recipe_subclass_shards(dump):
    """Recipe subclasses become sharded when total_shards > 1."""
    root, _, _ = dump

    class Recipe(OmniIODataset):
        def __init__(self, split, total_shards=None):
            super().__init__(
                query=PAIRED.format(where=""),
                fields={"speech": "audio", "text": "text"},
                params={"split": split},
                dump_root=root,
                total_shards=total_shards,
            )

    ds = Recipe("train", total_shards=2)
    assert isinstance(ds, Recipe) and isinstance(ds, ShardedDataset)
    assert len(ds.shard(0)) + len(ds.shard(1)) == len(ds)
    assert type(Recipe("train")) is Recipe


def test_errors(dump):
    """Bad queries, fields and dump arguments raise errors."""
    root, _, _ = dump
    with pytest.raises(ValueError, match="'id'"):
        OmniIODataset(query="SELECT 1 AS x", fields={"text": "text"})
    with pytest.raises(ValueError, match="neither"):
        OmniIODataset(query="SELECT 'a' AS id", fields={"text": "text"})
    with pytest.raises(ValueError, match="known kinds"):
        OmniIODataset(query="SELECT 'a' AS id", fields={"v": "video"})
    with pytest.raises(ValueError, match="at least one field"):
        OmniIODataset(query="SELECT 'a' AS id", fields={})
    with pytest.raises(ValueError):
        dump_blob(root / "a", "text", ["a"], ["1", "2"])


def test_query_trailing_semicolon(dump):
    """A trailing semicolon in the query is accepted."""
    root, _, _ = dump
    ds = OmniIODataset(
        query=(
            "SELECT id, id AS text FROM "
            "read_parquet('audio/toy/train/metadata.parquet') "
            "ORDER BY id ;  \n"
        ),
        fields={"text": "text"},
        dump_root=root,
    )
    assert len(ds) == N_AUDIO and ds[1]["text"] == "u01"


def test_dump_root_with_apostrophe(dump, tmp_path):
    """An apostrophe in the dump root is quoted in SQL literals."""
    root, _, _ = dump
    quoted = tmp_path / "it's"
    quoted.mkdir()
    (quoted / "audio").symlink_to(root / "audio")
    ds = OmniIODataset(
        query=(
            "SELECT id, id AS text FROM "
            "read_parquet('audio/toy/train/metadata.parquet')"
        ),
        fields={"text": "text"},
        dump_root=quoted,
    )
    assert len(ds) == N_AUDIO


def test_shard_by_column_with_quote(dump):
    """A shard_by name with a double quote is quoted as an identifier."""
    root, _, _ = dump
    query = PAIRED.format(where="").replace(
        "SELECT a.id,", 'SELECT a.id, a.id AS "g""x",'
    )
    ds = OmniIODataset(
        query=query,
        fields={"speech": "audio", "text": "text"},
        params={"split": "train"},
        dump_root=root,
        total_shards=2,
        shard_by='g"x',
    )
    assert len(ds.shard(0)) + len(ds.shard(1)) == len(ds)


class RecipeDataset(OmniIODataset):
    """A module-level recipe subclass, so pickle can find it by name."""

    def __init__(self, root, total_shards=None):
        """Select the paired toy data."""
        super().__init__(
            query=PAIRED.format(where=""),
            fields={"speech": "audio", "text": "text"},
            params={"split": "train"},
            dump_root=root,
            total_shards=total_shards,
        )


def test_sharded_datasets_pickle(dump):
    """Sharded datasets and their shards survive pickling (spawn workers)."""
    root, _, _ = dump
    for ds in (RecipeDataset(root, total_shards=2), _paired(root, total_shards=2)):
        for obj in (ds, ds.shard(1)):
            copy = pickle.loads(pickle.dumps(obj))
            assert type(copy) is type(obj)
            assert isinstance(copy, ShardedDataset)
            assert copy.keys() == obj.keys()
            assert copy[0]["text"] == obj[0]["text"]
    assert type(pickle.loads(pickle.dumps(RecipeDataset(root)))) is RecipeDataset


def test_resolve_dump_root_default(monkeypatch, tmp_path):
    """The default applies only when neither argument nor environment is set."""
    monkeypatch.delenv(DUMP_ROOT_ENV, raising=False)
    assert resolve_dump_root(None, default=tmp_path) == tmp_path
    monkeypatch.setenv(DUMP_ROOT_ENV, "/env")
    assert resolve_dump_root(None, default=tmp_path) == resolve_dump_root("/env")


def test_partial_dump_is_not_complete(tmp_path):
    """A failed append leaves metadata.parquet but no completion marker."""
    path = tmp_path / "text"
    files = []
    for i in range(4):
        files.append(tmp_path / f"t{i}.txt")
        files[-1].write_text(f"text {i}")
    files[2] = tmp_path / "missing.txt"
    ids = [f"t{i}" for i in range(4)]
    with pytest.raises(Exception):
        dump_blob(
            path,
            "text",
            [str(f) for f in files],
            ids,
            is_path=True,
            compression_level=3,
        )
    assert not is_blob_dumped(path)
    files[2] = tmp_path / "t2.txt"
    files[2].write_text("text 2")
    dump_blob(
        path, "text", [str(f) for f in files], ids, is_path=True, compression_level=3
    )
    assert is_blob_dumped(path) and (path / DUMP_COMPLETE_FILE).is_file()


def test_moved_dump_names_missing_blob(dump, tmp_path):
    """A moved dump fails with an error naming the stale blob path."""
    import shutil

    root, _, _ = dump
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    shutil.rmtree(root)
    ds = _paired(moved)
    with pytest.raises(FileNotFoundError, match="absolute blob paths"):
        ds[0]
    with pytest.raises(FileNotFoundError, match="absolute blob paths"):
        ds.keys()


def test_null_length_rejected():
    """NULL lengths (e.g. after a LEFT JOIN) give an actionable error."""
    ds = OmniIODataset(
        query=(
            "SELECT * FROM (VALUES ('a', 'x', 1), ('b', 'y', NULL)) "
            "t(id, text, length)"
        ),
        fields={"text": "text"},
    )
    with pytest.raises(ValueError, match="NULL for 1 rows"):
        ds.get_item_lengths()


def test_dump_root_with_comma_rejected(tmp_path):
    """Commas in the dump root are refused: file_search_path splits on them."""
    with pytest.raises(ValueError, match="comma"):
        OmniIODataset(
            query="SELECT 'a' AS id, 'x' AS text",
            fields={"text": "text"},
            dump_root=tmp_path / "a,b",
        )


def test_any_registered_kind_is_accepted():
    """Fields are not limited to audio/text: any ESPnet3 kind is accepted."""
    ds = OmniIODataset(
        query="SELECT * FROM (VALUES ('a', 'hi', 3)) t(id, text, spk)",
        fields={"text": "text", "spk": "number"},
    )
    assert ds[0] == {"text": "hi", "spk": 3}


def test_blob_modality_without_omniio_reader_fails(dump):
    """A blob field whose modality omniio cannot read raises on access."""
    root, _, _ = dump
    ds = OmniIODataset(
        query="SELECT 'a' AS id, 'x.bin' AS n_path, 0 AS n_start, 1 AS n_size",
        fields={"n": "number"},
    )
    with pytest.raises(ValueError, match="no reader for modality 'number'"):
        ds._read_blob("number", str(root / "x.bin"), 0, 1)
