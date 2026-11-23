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


def test_speed_perturbation_adds_a_perturbed_speaker_per_factor(tmp_path, utterances):
    _make_split(tmp_path, "voxceleb1_dev", utterances)
    dataset = VoxCelebDataset(
        split="voxceleb1_dev",
        recipe_dir=tmp_path,
        speed_perturb_factors=[0.9, 1.0, 1.1],
    )

    assert len(dataset) == 9
    # Factor-major: the three utterances at 0.9, then at 1.0, then at 1.1.
    for copy, (factor, prefix) in enumerate(
        [(0.9, "sp0.9-"), (1.0, ""), (1.1, "sp1.1-")]
    ):
        for row, (_, speaker, samples) in enumerate(sorted(utterances)):
            item = dataset[copy * 3 + row]
            assert item["spk_labels"] == prefix + speaker
            assert item["speech"].dtype == np.float32
            # `sox speed f` makes the signal 1/f as long.
            assert len(item["speech"]) == pytest.approx(len(samples) / factor, abs=1)


def test_speed_factor_one_is_the_original_utterance(tmp_path, utterances):
    _make_split(tmp_path, "voxceleb1_dev", utterances)
    plain = VoxCelebDataset(split="voxceleb1_dev", recipe_dir=tmp_path)
    perturbed = VoxCelebDataset(
        split="voxceleb1_dev", recipe_dir=tmp_path, speed_perturb_factors=[0.9, 1.0]
    )

    np.testing.assert_array_equal(perturbed[3]["speech"], plain[0]["speech"])


def test_speed_perturbation_scales_frequency(tmp_path):
    tone = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)
    _make_split(tmp_path, "voxceleb1_dev", [("id10001/v/1", "id10001", tone)])
    dataset = VoxCelebDataset(
        split="voxceleb1_dev", recipe_dir=tmp_path, speed_perturb_factors=[1.1]
    )

    speech = dataset[0]["speech"]
    peak_hz = np.argmax(np.abs(np.fft.rfft(speech))) * 16000 / len(speech)
    assert peak_hz == pytest.approx(1100, abs=5)


def test_slowed_down_audio_leaves_the_top_band_empty_like_sox():
    # Slowing down by 0.9 maps 8 kHz to 7.2 kHz, so the band above it must stay
    # empty. On white noise sox / libsoxr leave it >100 dB down; scipy's default
    # resample_poly filter left it only ~16 dB down, and a model trained on
    # sox-perturbed copies does not recognize such copies.
    from egs3.voxceleb.esp2_spk.dataset.dataset import speed_perturb

    noise = np.random.default_rng(0).standard_normal(16000 * 4).astype(np.float32)
    slowed = speed_perturb(noise, 0.9)
    # Hann window, so leakage from the passband does not mask the stopband.
    power = np.abs(np.fft.rfft(slowed * np.hanning(len(slowed)))) ** 2
    hz = np.fft.rfftfreq(len(slowed), 1 / 16000)

    passband = power[(hz > 300) & (hz < 6500)].mean()
    stopband = power[hz > 7500].mean()
    assert 10 * np.log10(stopband / passband) < -60


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speed_perturb_factors": []},
        {"speed_perturb_factors": [1.0, 1.0]},
        {"speed_perturb_factors": [0.0]},
        {"speed_perturb_factors": [0.9, 1.0], "trials": "vox1_o"},
    ],
)
def test_invalid_speed_perturbation_is_rejected(tmp_path, utterances, kwargs):
    _make_split(tmp_path, "voxceleb1_dev", utterances)
    with pytest.raises(ValueError, match="speed_perturb_factors"):
        VoxCelebDataset(split="voxceleb1_dev", recipe_dir=tmp_path, **kwargs)


def _speaker_split(tmp_path):
    rng = np.random.default_rng(0)
    utts = [
        (f"id1{s:04d}/v/{u}", f"id1{s:04d}", rng.uniform(-0.5, 0.5, 160))
        for s in range(8)
        for u in range(s + 1)  # unbalanced: 1..8 utterances per speaker
    ]
    _make_split(tmp_path, "voxceleb1_dev", utts)
    return VoxCelebDataset(
        split="voxceleb1_dev", recipe_dir=tmp_path, speed_perturb_factors=[0.9, 1.0]
    )


def _assert_balanced(batches, labels, batch_size, n_indices):
    # Only the last batch may repeat a speaker.
    for batch in batches[:-1]:
        assert len({labels[i] for i in batch}) == len(batch) == batch_size
    used = [i for batch in batches for i in batch]
    assert len(used) == len(set(used)) == n_indices // batch_size * batch_size


def test_speaker_balanced_batches_use_every_utterance(tmp_path):
    import torch
    from torch.utils.data import RandomSampler

    from egs3.voxceleb.esp2_spk.dataset.sampler import SpeakerBalancedBatchSampler

    dataset = _speaker_split(tmp_path)
    labels = dataset.speaker_labels()
    assert labels == [dataset[i]["spk_labels"] for i in range(len(dataset))]

    for seed in range(10):
        sampler = SpeakerBalancedBatchSampler(
            RandomSampler(dataset, generator=torch.Generator().manual_seed(seed)),
            batch_size=8,
            drop_last=True,
            dataset=dataset,
        )
        batches = list(sampler)
        assert len(batches) == len(sampler) == len(dataset) // 8
        _assert_balanced(batches, labels, 8, len(dataset))


def test_lightning_shards_speaker_balanced_batches_under_ddp(tmp_path):
    """Lightning rebuilds the sampler around a DistributedSampler per rank."""
    from lightning.fabric.utilities.data import _replace_dunder_methods
    from lightning.pytorch.utilities.data import _update_dataloader
    from torch.utils.data import (
        BatchSampler,
        DataLoader,
        DistributedSampler,
        RandomSampler,
    )

    from egs3.voxceleb.esp2_spk.dataset.sampler import SpeakerBalancedBatchSampler

    dataset = _speaker_split(tmp_path)
    labels = dataset.speaker_labels()
    # As Lightning's data connector does around `train_dataloader()`.
    with (
        _replace_dunder_methods(DataLoader, "dataset"),
        _replace_dunder_methods(BatchSampler),
    ):
        loader = DataLoader(
            dataset,
            batch_sampler=SpeakerBalancedBatchSampler(
                RandomSampler(dataset), batch_size=4, drop_last=True, dataset=dataset
            ),
        )

    shards = []
    for rank in range(2):
        shard = DistributedSampler(dataset, num_replicas=2, rank=rank, seed=0)
        shard.set_epoch(3)
        rebuilt = _update_dataloader(loader, shard)
        assert isinstance(rebuilt.batch_sampler, SpeakerBalancedBatchSampler)
        assert rebuilt.batch_sampler.sampler is shard
        batches = list(rebuilt.batch_sampler)
        assert len(batches) == len(rebuilt.batch_sampler) == len(shard) // 4
        _assert_balanced(batches, labels, 4, len(shard))
        shards.append({i for batch in batches for i in batch})
    assert not shards[0] & shards[1]
