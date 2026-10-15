import numpy as np
import pytest
import soundfile as sf

from egs3.voxceleb.esp2_spk.dataset import builder
from egs3.voxceleb.esp2_spk.dataset.dataset import VoxCelebDataset


def _make_split(recipe_dir, split, utterances):
    """Write WAVs plus the manifests and archive `create_dataset` would."""
    data_root = recipe_dir / "data"
    entries = []
    for utt_id, speaker, samples in utterances:
        path = recipe_dir / "audio" / f"{utt_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, samples, 16000, subtype="PCM_16")
        entries.append((utt_id, speaker, path))
    builder.write_manifests(data_root / split, entries)
    builder.pack_split(builder.blob_dir(data_root, split), entries, n_workers=0)
    return data_root / split


@pytest.fixture
def utterances():
    rng = np.random.default_rng(0)
    return [
        (f"id1000{i}/vid/0000{i}", f"id1000{i}", rng.uniform(-0.5, 0.5, 1600 * (i + 1)))
        for i in range(3)
    ]


def test_utterances_are_read_back_from_the_archive(tmp_path, utterances):
    _make_split(tmp_path, "voxceleb1_dev", utterances)
    dataset = VoxCelebDataset(split="voxceleb1_dev", recipe_dir=tmp_path)

    assert len(dataset) == 3
    for index, (utt_id, speaker, samples) in enumerate(sorted(utterances)):
        item = dataset[index]
        assert item["spk_labels"] == speaker
        assert item["speech"].dtype == np.float32
        # Packed as WAV, so the archive decodes to exactly what the file held.
        np.testing.assert_array_equal(
            item["speech"],
            sf.read(tmp_path / "audio" / f"{utt_id}.wav", dtype="float32")[0],
        )


def test_trials_read_both_sides_and_declare_both(tmp_path, utterances):
    split_dir = _make_split(tmp_path, "voxceleb1_test", utterances)
    (split_dir / "vox1_o.trials").write_text(
        f"1 {utterances[0][0]} {utterances[1][0]}\n", encoding="utf-8"
    )
    dataset = VoxCelebDataset(
        split="voxceleb1_test", trials="vox1_o", recipe_dir=tmp_path
    )

    item = dataset[0]
    assert [f.name for f in dataset.fields] == ["speech", "speech2", "spk_labels"]
    assert len(item["speech"]) == 1600 and len(item["speech2"]) == 3200
    assert item["spk_labels"] == "1"


def test_pack_split_resumes_without_duplicating(tmp_path, utterances):
    split_dir = _make_split(tmp_path, "voxceleb1_dev", utterances)
    entries = [
        (utt_id, speaker, tmp_path / "audio" / f"{utt_id}.wav")
        for utt_id, speaker, _ in utterances
    ]
    builder.pack_split(
        builder.blob_dir(split_dir.parent, "voxceleb1_dev"), entries, n_workers=0
    )
    assert builder.blob_rows(split_dir / "blob") == 3


def test_a_missing_archive_points_at_create_dataset(tmp_path, utterances):
    split_dir = _make_split(tmp_path, "voxceleb1_dev", utterances)
    (split_dir / "blob" / "metadata.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="create_dataset"):
        VoxCelebDataset(split="voxceleb1_dev", recipe_dir=tmp_path)


def test_data_dir_reads_a_relocated_copy(tmp_path, utterances):
    split_dir = _make_split(tmp_path / "recipe", "voxceleb1_dev", utterances)
    staged = tmp_path / "staged"
    split_dir.parent.rename(staged)

    dataset = VoxCelebDataset(
        split="voxceleb1_dev", recipe_dir=tmp_path / "recipe", data_dir=staged
    )

    assert len(dataset[0]["speech"]) == 1600
