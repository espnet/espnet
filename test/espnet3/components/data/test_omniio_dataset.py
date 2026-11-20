"""Tests for espnet3.components.data.omniio_dataset and its dataloader hooks."""

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

pytest.importorskip("omniio")
pytest.importorskip("duckdb")

from espnet3.components.data.dataloader import (  # noqa: E402
    DataLoaderBuilder,
    _dataset_shape_file,
)
from espnet3.components.data.dataset import (  # noqa: E402
    CombinedDataset,
    ShardedDataset,
)
from espnet3.components.data.omniio_dataset import (  # noqa: E402
    DUMP_ROOT_ENV,
    OmniIODataset,
    ShardedOmniIODataset,
    blob_dir,
    dump_blob,
    is_blob_dumped,
    render_query,
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
FROM read_parquet('{dump_root}/audio/toy/{split}/metadata.parquet') AS a
JOIN read_parquet('{dump_root}/text/toy/{split}/metadata.parquet') AS t USING (id)
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
        blob_dir(root, "audio", "toy", "train"),
        "audio",
        [str(tmp_path / f"{uid}.flac") for uid in audio],
        list(audio),
        num_workers=2,
    )
    texts = {uid: f"TEXT {uid}" for uid in list(audio)[::2]}
    texts["orphan"] = "NO AUDIO"
    dump_blob(
        blob_dir(root, "text", "toy", "train"),
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
        query=PAIRED,
        fields={"speech": "audio", "text": "text"},
        params={"split": "train", "where": where},
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
    assert blob_dir(root, "audio", "toy", "train") == root / "audio" / "toy" / "train"
    assert is_blob_dumped(root / "audio" / "toy" / "train")
    assert not is_blob_dumped(root / "audio" / "toy" / "missing")


def test_render_query():
    """Only {identifier} placeholders are substituted."""
    sql = "SELECT {'a': 1} FROM '{root}/{split}.parquet'"
    assert render_query(sql, {"root": "/d", "split": "x"}) == (
        "SELECT {'a': 1} FROM '/d/x.parquet'"
    )
    with pytest.raises(KeyError, match="split"):
        render_query(sql, {"root": "/d"})


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
        ds.item_lengths(), [len(audio[uid]) for uid in paired]
    )
    assert ds["u04"]["text"] == "TEXT u04"


def test_filter_in_query(dump):
    """A WHERE clause in the query filters rows."""
    root, audio, _ = dump
    ds = _paired(root, where=f"WHERE a.samples > {len(audio['u05'])}")
    assert ds.keys() == ["u06", "u08", "u10"]


def test_inline_column(dump):
    """Inline columns are used as-is, with transforms applied."""
    root, _, _ = dump
    ds = OmniIODataset(
        query=(
            "SELECT id, upper(id) AS text, 1 AS length FROM "
            "read_parquet('{dump_root}/audio/toy/train/metadata.parquet') "
            "ORDER BY id"
        ),
        fields={"text": "text"},
        dump_root=root,
        transforms={"text": str.lower},
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
    query = PAIRED.replace(
        "SELECT a.id,", "SELECT a.id, CAST(substr(a.id, 2) AS INT) % 3 AS grp,"
    )
    ds = OmniIODataset(
        query=query,
        fields={"speech": "audio", "text": "text"},
        params={"split": "train", "where": ""},
        dump_root=root,
        total_shards=2,
        shard_by="grp",
    )
    import duckdb

    full = duckdb.sql(ds.query).fetchall()
    grp = dict(duckdb.sql(f"SELECT id, grp FROM ({ds.query})").fetchall())
    expected = dict(
        duckdb.sql(f"SELECT id, hash(grp) % 2 FROM ({ds.query})").fetchall()
    )
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
                query=PAIRED,
                fields={"speech": "audio", "text": "text"},
                params={"split": split, "where": ""},
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
    with pytest.raises(ValueError, match="modality"):
        OmniIODataset(query="SELECT 'a' AS id", fields={"v": "video"})
    with pytest.raises(ValueError):
        dump_blob(root / "a", "text", ["a"], ["1", "2"])


# ---------------------------------------------------------------------- #
# DataLoaderBuilder: lengths_from_dataset and rank-local shards
# ---------------------------------------------------------------------- #


def _combined(ds):
    combined = CombinedDataset([ds], [(None, None)], use_espnet_preprocessor=False)
    combined.use_espnet_collator = True  # items come as (uid, item)
    return combined


def collate_texts(batch):
    """Collate a batch of ``(uid, item)`` pairs to its transcripts."""
    return [item["text"] for _, item in batch]


def test_dataset_shape_file(dump):
    """The temporary shape file lists every item's length by index."""
    root, _, _ = dump
    combined = _combined(_paired(root))
    with _dataset_shape_file(combined) as path:
        lines = open(path).read().splitlines()
    expected = combined.datasets[0].item_lengths()
    assert lines == [f"{i} {n}" for i, n in enumerate(expected)]


def _iter_config(batch_bins):
    """Return a dataloader config batching by lengths_from_dataset."""
    return OmegaConf.create(
        {
            "dataloader": {
                "train": {
                    "iter_factory": {
                        "_target_": (
                            "espnet2.iterators.sequence_iter_factory."
                            "SequenceIterFactory"
                        ),
                        "shuffle": False,
                        "collate_fn": {
                            "_target_": (
                                "test.espnet3.components.data."
                                "test_omniio_dataset.collate_texts"
                            ),
                            "_partial_": True,
                        },
                        "batches": {
                            "type": "numel",
                            "lengths_from_dataset": True,
                            "shape_files": [],
                            "batch_size": 1,
                            "batch_bins": batch_bins,
                        },
                    }
                }
            }
        }
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_builder_rank_shards_are_not_strided(monkeypatch, dump, rank):
    """Each rank batches only its own shard and matches batch counts."""
    import espnet3.components.data.dataloader as dl

    root, _, _ = dump
    monkeypatch.setattr(dl.torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(dl.torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(dl.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(dl.torch.distributed, "get_backend", lambda: "gloo")
    # The other rank has 2 batches: this rank must keep exactly 2.
    monkeypatch.setattr(
        dl.torch.distributed,
        "all_reduce",
        lambda t, op=None: t.fill_(min(int(t.item()), 2)),
    )
    sharded = _paired(root, total_shards=2, dist_world_size=2)
    combined = _combined(sharded)
    builder = DataLoaderBuilder(
        dataset=combined,
        config=_iter_config(1),  # one utterance per batch
        collate_fn=None,
        num_device=2,
        epoch=0,
    )
    loader = builder.build("train")
    assert builder._rank_sharded
    batches = list(loader)
    assert len(batches) == 2
    shard_texts = {f"TEXT {uid}" for uid in sharded.shard(rank).keys()}
    emitted = [text for batch in batches for text in batch]
    assert len(emitted) == 2 and set(emitted) <= shard_texts


def test_query_trailing_semicolon(dump):
    """A trailing semicolon in the query is accepted."""
    root, _, _ = dump
    ds = OmniIODataset(
        query=(
            "SELECT id, id AS text FROM "
            "read_parquet('{dump_root}/audio/toy/train/metadata.parquet') "
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
            "read_parquet('{dump_root}/audio/toy/train/metadata.parquet')"
        ),
        fields={"text": "text"},
        dump_root=quoted,
    )
    assert len(ds) == N_AUDIO


def test_shard_by_column_with_quote(dump):
    """A shard_by name with a double quote is quoted as an identifier."""
    root, _, _ = dump
    query = PAIRED.replace("SELECT a.id,", 'SELECT a.id, a.id AS "g""x",')
    ds = OmniIODataset(
        query=query,
        fields={"speech": "audio", "text": "text"},
        params={"split": "train", "where": ""},
        dump_root=root,
        total_shards=2,
        shard_by='g"x',
    )
    assert len(ds.shard(0)) + len(ds.shard(1)) == len(ds)


def test_first_item_transform_runs_every_time(dump):
    """The cached first row is decoded and transformed on every read."""
    root, _, _ = dump
    calls = []
    ds = OmniIODataset(
        query=(
            "SELECT id, id AS text FROM "
            "read_parquet('{dump_root}/audio/toy/train/metadata.parquet') "
            "ORDER BY id"
        ),
        fields={"text": "text"},
        dump_root=root,
        transforms={"text": lambda t: calls.append(t) or t.upper()},
    )
    assert ds[0]["text"] == "U00" and ds[0]["text"] == "U00"
    assert ds._index is None and calls == ["u00", "u00"]


def test_duplicate_ids_rejected_by_id_lookup(dump):
    """Looking up by ID fails clearly when IDs are duplicated."""
    root, _, _ = dump
    ds = OmniIODataset(
        query=(
            "SELECT 'dup' AS id, id AS text FROM "
            "read_parquet('{dump_root}/audio/toy/train/metadata.parquet')"
        ),
        fields={"text": "text"},
        dump_root=root,
    )
    assert len(ds) == N_AUDIO  # integer access still works
    with pytest.raises(ValueError, match="duplicate IDs"):
        ds.get_by_id("dup")
