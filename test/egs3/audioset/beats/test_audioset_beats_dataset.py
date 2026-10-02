"""Tests for the AudioSet-2M BEATs dataset builder and dataset."""

import builtins
import json

import numpy as np
import pytest
import soundfile as sf

import espnet3.parallel.parallel as parallel_module
from egs3.audioset.beats.dataset import Dataset, DatasetBuilder
from egs3.audioset.beats.dataset.builder import (
    AudioSetExample,
    PrepareClipRunner,
    _prepare_clip,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                    | Description                    |
# |----------------------------------------------|--------------------------------|
# | test_builder_writes_filtered_manifests       | Cuts short segments, drops     |
# |                                              | missing/corrupt/long clips,    |
# |                                              | keeps stable utterance ids.    |
# | test_builder_requires_source                 | Missing AudioSet root says it  |
# |                                              | must be downloaded.            |
# | test_prepare_clip_runner_merges_in_idx_order | Shard results come back in     |
# |                                              | clip-list order.               |
# | test_prepare_clip_drops_unreadable_source    | Corrupt source clip is dropped.|
# | test_prepare_clip_propagates_write_errors    | A failed cut stops the build   |
# |                                              | instead of dropping the clip.  |
# | test_dataset_reads_waveforms_and_targets     | Raw waveform + target join.    |
# | test_dataset_reads_kaldi_features            | feats_path mode.               |
# | test_dataset_rejects_unknown_split           | Unknown split raises.          |
# | test_dataset_requires_kaldiio_for_features   | feats_path without kaldiio     |
# |                                              | raises a pointed ImportError.  |

SAMPLE_RATE = 16000


def _write_wav(path, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = 0.1 * np.ones(int(seconds * SAMPLE_RATE), dtype=np.float32)
    sf.write(str(path), audio, SAMPLE_RATE)


def _write_csv(path, rows):
    header = "# Segments csv\n# YTID, start_seconds, end_seconds, positive_labels\n"
    body = "".join(f'{yt_id}, {start}, {end}, "/m/0"\n' for yt_id, start, end in rows)
    path.write_text(header + body, encoding="utf-8")


@pytest.fixture
def audioset_root(tmp_path):
    root = tmp_path / "audioset"
    root.mkdir()
    _write_csv(root / "eval_segments.csv", [("e0", 0.0, 10.0)])
    _write_wav(root / "eval_wav/e0.wav", 10.0)
    _write_csv(
        root / "unbalanced_train_segments.csv",
        [
            ("u0", 30.0, 40.0),  # kept as-is
            ("u1", 0.0, 4.0),  # cut to 4 s
            ("missing", 0.0, 10.0),  # not downloaded
            ("u2", 0.0, 10.0),  # 12 s file, dropped by max_wav_duration
        ],
    )
    _write_wav(root / "unbalanced_wav/u0.wav", 10.0)
    _write_wav(root / "unbalanced_wav/u1.wav", 10.0)
    _write_wav(root / "unbalanced_wav/u2.wav", 12.0)
    _write_csv(root / "balanced_train_segments.csv", [("b0", 0.0, 10.0)])
    (root / "balance_wav").mkdir()
    (root / "balance_wav/b0.wav").write_bytes(b"corrupt")
    return root


def _read_manifest(path):
    return [line.split("\t") for line in path.read_text().splitlines()]


@pytest.fixture(autouse=True)
def no_parallel(monkeypatch):
    """Run PrepareClipRunner on the driver, whatever other tests configured."""
    monkeypatch.setattr(parallel_module, "parallel_config", None)


def test_builder_writes_filtered_manifests(tmp_path, audioset_root):
    recipe_dir = tmp_path / "recipe"
    builder = DatasetBuilder()
    assert builder.is_source_prepared(recipe_dir, source_dir=audioset_root)
    assert not builder.is_built(recipe_dir)

    builder.build(recipe_dir, source_dir=audioset_root)

    assert builder.is_built(recipe_dir)
    train = _read_manifest(recipe_dir / "data/manifest/train.tsv")
    # Ids number downloaded clips in segment-list order: u0, u1, u2, b0.
    assert [row[0] for row in train] == ["as2m_20k-AudioSet-0", "as2m_20k-AudioSet-1"]
    assert train[0][1].endswith("unbalanced_wav/u0.wav")
    assert train[1][1] == str(recipe_dir.resolve() / "data/cut_wav/u1.wav")
    assert int(train[1][2]) == 4 * SAMPLE_RATE
    assert not (audioset_root / "cut_wav").exists()
    eval_rows = _read_manifest(recipe_dir / "data/manifest/eval.tsv")
    assert [row[0] for row in eval_rows] == ["as2m_20k-eval-0"]


def test_builder_requires_source(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIOSET", raising=False)
    builder = DatasetBuilder()

    assert not builder.is_source_prepared(tmp_path)
    with pytest.raises(FileNotFoundError, match="AudioSet was not found.*AUDIOSET"):
        builder.prepare_source(tmp_path)
    with pytest.raises(FileNotFoundError, match="download it first"):
        builder.prepare_source(tmp_path, source_dir=tmp_path / "missing")


def test_prepare_clip_runner_merges_in_idx_order(tmp_path):
    for shard, idxs in (("split.0", [2, 0]), ("split.1", [1])):
        shard_dir = tmp_path / shard
        shard_dir.mkdir()
        (shard_dir / "results.jsonl").write_text(
            "".join(
                json.dumps({"idx": i, "ok": True, "num_samples": i}) + "\n"
                for i in idxs
            )
        )

    records = PrepareClipRunner(provider=None).merge(
        [tmp_path / "split.1", tmp_path / "split.0"]
    )

    assert [record["idx"] for record in records] == [0, 1, 2]


def test_prepare_clip_drops_unreadable_source(tmp_path):
    source = tmp_path / "corrupt.wav"
    source.write_bytes(b"not audio")
    example = AudioSetExample(source, tmp_path / "cut.wav", 4.0)

    assert _prepare_clip(example) == (False, 0)


def test_prepare_clip_propagates_write_errors(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    _write_wav(source, 10.0)
    example = AudioSetExample(source, tmp_path / "cut" / "cut.wav", 4.0)

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sf, "write", full_disk)

    # Dropping the clip here would publish a manifest that is quietly short.
    with pytest.raises(OSError, match="No space left"):
        _prepare_clip(example)


def _write_manifest(recipe_dir, split, rows):
    path = recipe_dir / f"data/manifest/{split}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{u}\t{p}\t{n}\n" for u, p, n in rows), encoding="utf-8")


def test_dataset_reads_waveforms_and_targets(tmp_path):
    _write_wav(tmp_path / "a.wav", 1.0)
    _write_wav(tmp_path / "b.wav", 0.5)
    _write_manifest(
        tmp_path,
        "train",
        [("a", tmp_path / "a.wav", 16000), ("b", tmp_path / "b.wav", 8000)],
    )
    target_path = tmp_path / "target.scp"
    target_path.write_text("1 7 8\n0 5\n", encoding="utf-8")

    dataset = Dataset("train", recipe_dir=tmp_path, target_path=target_path)

    assert len(dataset) == 2
    assert dataset[1]["speech"].shape == (8000,)
    assert dataset[1]["speech"].dtype == np.float32
    assert dataset[0]["target"] == "5"
    assert dataset[1]["target"] == "7 8"
    assert set(Dataset("train", recipe_dir=tmp_path)[0]) == {"speech"}


def test_dataset_reads_kaldi_features(tmp_path):
    kaldiio = pytest.importorskip("kaldiio")
    feats = {"a": np.ones((98, 128), dtype=np.float32)}
    with kaldiio.WriteHelper(
        f"ark,scp:{tmp_path}/feats.ark,{tmp_path}/feats.scp"
    ) as writer:
        for key, value in feats.items():
            writer[key] = value
    _write_manifest(tmp_path, "eval", [("a", "-", 98)])

    dataset = Dataset("eval", recipe_dir=tmp_path, feats_path=tmp_path / "feats.scp")

    np.testing.assert_array_equal(dataset[0]["speech"], feats["a"])


def test_dataset_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="Unknown split"):
        Dataset("dev", recipe_dir=tmp_path)


def test_dataset_requires_kaldiio_for_features(tmp_path, monkeypatch):
    _write_manifest(tmp_path, "eval", [("a", "-", 98)])
    real_import = builtins.__import__

    def without_kaldiio(name, *args, **kwargs):
        if name == "kaldiio":
            raise ImportError("No module named 'kaldiio'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_kaldiio)

    with pytest.raises(ImportError, match="espnet\\[kaldiio\\]"):
        Dataset("eval", recipe_dir=tmp_path, feats_path=tmp_path / "feats.scp")

    # The default waveform path stays importable without kaldiio.
    assert len(Dataset("eval", recipe_dir=tmp_path)) == 1
