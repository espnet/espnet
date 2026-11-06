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
    return OmniIODataset(
        query=PAIRED,
        fields={"speech": "audio", "text": "text"},
        params={"split": "train", "where": where},
        dump_root=root,
        **kwargs,
    )


def test_resolve_dump_root(monkeypatch, tmp_path):
    monkeypatch.delenv(DUMP_ROOT_ENV, raising=False)
    with pytest.raises(ValueError):
        resolve_dump_root()
    monkeypatch.setenv(DUMP_ROOT_ENV, str(tmp_path))
    assert resolve_dump_root() == tmp_path


def test_blob_layout(dump):
    root, _, _ = dump
    assert blob_dir(root, "audio", "toy", "train") == root / "audio" / "toy" / "train"
    assert is_blob_dumped(root / "audio" / "toy" / "train")
    assert not is_blob_dumped(root / "audio" / "toy" / "missing")


def test_render_query():
    sql = "SELECT {'a': 1} FROM '{root}/{split}.parquet'"
    assert render_query(sql, {"root": "/d", "split": "x"}) == (
        "SELECT {'a': 1} FROM '/d/x.parquet'"
    )
    with pytest.raises(KeyError, match="split"):
        render_query(sql, {"root": "/d"})


def test_join_returns_paired_items(dump):
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
    root, audio, _ = dump
    ds = _paired(root, where=f"WHERE a.samples > {len(audio['u05'])}")
    assert ds.keys() == ["u06", "u08", "u10"]


def test_inline_column(dump):
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
    root, _, texts = dump
    ds = _paired(root)
    assert len(ds) == 6 and ds._index is None  # COUNT query only
    assert ds[0]["text"] == "TEXT u00" and ds._index is None  # LIMIT 1 probe
    ds[1]
    assert ds._index is not None


def test_unsharded_has_no_shard_metadata(dump):
    root, _, _ = dump
    for total in (None, 1):
        ds = _paired(root, total_shards=total)
        assert type(ds) is OmniIODataset
        assert not isinstance(ds, ShardedDataset)
        assert ds.total_shards is None and ds.dist_world_size is None


def test_shards_partition_rows(dump):
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
    root, _, _ = dump
    query = PAIRED.replace("SELECT a.id,", "SELECT a.id, a.samples % 2 AS grp,")
    ds = OmniIODataset(
        query=query,
        fields={"speech": "audio", "text": "text"},
        params={"split": "train", "where": ""},
        dump_root=root,
        total_shards=2,
        shard_by="grp",
    )
    groups = [ds.shard(i)._index for i in range(2)]
    assert sum(len(g["id"]) for g in groups) == len(ds)


def test_recipe_subclass_shards(dump):
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


def collate_uids(batch):
    """Collate a batch to its uids."""
    return [uid for uid, _ in batch]


def test_dataset_shape_file(dump):
    root, _, _ = dump
    combined = _combined(_paired(root))
    with _dataset_shape_file(combined) as path:
        lines = open(path).read().splitlines()
    expected = combined.datasets[0].item_lengths()
    assert lines == [f"{i} {n}" for i, n in enumerate(expected)]


def _iter_config(batch_bins):
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
                                "test_omniio_dataset.collate_uids"
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


def test_builder_batches_from_lengths(dump):
    root, _, _ = dump
    combined = _combined(_paired(root))
    lengths = combined.datasets[0].item_lengths()
    builder = DataLoaderBuilder(
        dataset=combined,
        config=_iter_config(int(lengths.max()) * 2),
        collate_fn=None,
        num_device=1,
        epoch=0,
    )
    loader = builder.build("train")
    seen = sorted(int(u) for batch in loader for u in batch)
    assert seen == list(range(len(lengths)))


@pytest.mark.parametrize("rank", [0, 1])
def test_builder_rank_shards_are_not_strided(monkeypatch, dump, rank):
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
    shard = sharded.shard(rank)
    assert all(int(u) < len(shard) for batch in batches for u in batch)
