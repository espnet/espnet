"""Tests for the AudioSet-2M BEATs dataset builder and dataset."""

import kaldiio
import numpy as np
import pytest
import soundfile as sf

from egs3.audioset.ssl.dataset import Dataset, DatasetBuilder

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                    | Description                    |
# |----------------------------------------------|--------------------------------|
# | test_builder_writes_filtered_manifests       | Cuts short segments, drops     |
# |                                              | missing/corrupt/long clips,    |
# |                                              | keeps stable utterance ids.    |
# | test_builder_requires_source                 | Missing AudioSet root.         |
# | test_dataset_reads_waveforms_and_targets     | Raw waveform + target join.    |
# | test_dataset_reads_kaldi_features            | feats_path mode.               |
# | test_dataset_rejects_unknown_split           | Unknown split raises.          |

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


def test_builder_writes_filtered_manifests(tmp_path, audioset_root):
    recipe_dir = tmp_path / "recipe"
    builder = DatasetBuilder()
    assert builder.is_source_prepared(recipe_dir, source_dir=audioset_root)
    assert not builder.is_built(recipe_dir)

    builder.build(recipe_dir, source_dir=audioset_root, num_workers=1)

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
    with pytest.raises(FileNotFoundError, match="AUDIOSET"):
        builder.prepare_source(tmp_path)


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
