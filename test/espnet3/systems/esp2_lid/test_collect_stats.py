"""Tests for model-free LID statistics and global Dataset IDs."""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

import espnet3.systems.esp2_lid.collect_stats as stats_module
from espnet2.fileio.read_text import read_2columns_text
from espnet2.samplers.build_batch_sampler import build_category_batch_sampler


@pytest.mark.parametrize("empty_mode", ["train", "valid"])
def test_collect_speech_shapes_rejects_empty_dataset(tmp_path, monkeypatch, empty_mode):
    """Reject empty splits before the runner writes partial outputs."""
    organizer = SimpleNamespace(train=[{}], valid=[{}])
    setattr(organizer, empty_mode, [])
    monkeypatch.setattr(stats_module, "instantiate", lambda config: organizer)
    config = OmegaConf.create(
        {"dataset": {}, "dataloader": {}, "stats_dir": str(tmp_path)}
    )

    with pytest.raises(ValueError, match=f"{empty_mode} dataset is empty"):
        stats_module.collect_speech_shapes(config)

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("num_workers", [1, 2])
def test_combined_category_metadata_matches_shapes(tmp_path, monkeypatch, num_workers):
    """Use global IDs for real collection and sampling across two source datasets."""
    # Exercise real Dask workers without submitting nested scheduler jobs.
    import espnet3.parallel.parallel as parallel_module

    if num_workers > 1:
        distributed = pytest.importorskip("distributed")
        Client, LocalCluster = distributed.Client, distributed.LocalCluster

        monkeypatch.setattr(
            parallel_module,
            "_build_client",
            lambda config: Client(
                LocalCluster(
                    n_workers=2,
                    threads_per_worker=1,
                    processes=True,
                    dashboard_address=None,
                )
            ),
        )
    entries = []
    for source_index in range(2):
        source = tmp_path / f"source{source_index}"
        (source / "train").mkdir(parents=True)
        rows = []
        for index in range(4):
            wave = source / f"{index}.wav"
            sf.write(wave, np.zeros(160 + 10 * (source_index * 4 + index)), 16000)
            language = "eng" if index % 2 == 0 else "jpn"
            rows.append(f"{index}\t{wave}\t{language}\n")
        (source / "train/manifest.tsv").write_text("".join(rows), encoding="utf-8")
        entries.append(
            {
                "data_src": "egs3.voxlingua107.esp2_lid.dataset",
                "data_src_args": {
                    "data_dir": str(source),
                    "split": "train",
                },
            }
        )
    config = OmegaConf.create(
        {
            "dataset": {
                "_target_": "espnet3.components.data.data_organizer.DataOrganizer",
                "_recursive_": False,
                "train": entries,
                "valid": entries[::-1],
                "preprocessor": {"_target_": "must.not.be.instantiated"},
            },
            "dataloader": {
                mode: {"iter_factory": {"num_workers": num_workers}}
                for mode in ("train", "valid")
            },
            "stats_dir": str(tmp_path / "stats"),
            "parallel": {
                "env": "pbs" if num_workers > 1 else "local",
                "n_workers": num_workers,
            },
        }
    )
    stats_module.collect_speech_shapes(config)

    for mode in ("train", "valid"):
        output_dir = tmp_path / "stats" / mode
        expected = "eng 0 2 4 6\njpn 1 3 5 7\n"
        assert (output_dir / "category2utt").read_text() == expected
        assert (output_dir / "lang2utt").read_text() == expected
        assert (output_dir / "dataset2utt").read_text() == "0 0 1 2 3\n1 4 5 6 7\n"
        assert read_2columns_text(output_dir / "utt2dataset") == {
            str(index): str(index // 4) for index in range(8)
        }
        shapes = read_2columns_text(output_dir / "speech_shape")
        order = list(range(8)) if mode == "train" else [4, 5, 6, 7, 0, 1, 2, 3]
        assert shapes == {
            str(index): str(160 + 10 * original) for index, original in enumerate(order)
        }
        for batch_type in ("catbel", "catpow", "catpow_balance_dataset"):
            sampler, _ = build_category_batch_sampler(
                type=batch_type,
                category2utt_file=str(output_dir / "category2utt"),
                dataset2utt_parent_dir=str(output_dir),
                shape_files=[str(output_dir / "speech_shape")],
                batch_size=2,
                batch_bins=1000,
                min_batch_size=1,
                max_batch_size=2,
                upsampling_factor=0.5,
                category_upsampling_factor=0.5,
                dataset_upsampling_factor=0.5,
                dataset_scaling_factor=10.0,
                epoch=1,
            )
            sampled = {int(index) for batch in sampler for index in batch}
            assert sampled & {0, 1, 2, 3}, batch_type
            assert sampled & {4, 5, 6, 7}, batch_type
            assert sampled <= set(range(8)), batch_type

    # Replacing two datasets with one must discard stale global IDs.
    config.dataset.train = [entries[0]]
    config.dataset.valid = [entries[0]]
    stats_module.collect_speech_shapes(config)
    assert (tmp_path / "stats/train/category2utt").read_text() == "eng 0 2\njpn 1 3\n"


def test_collect_speech_shapes_rejects_non_string_language(tmp_path, monkeypatch):
    """Fail clearly rather than writing unusable category mappings."""
    samples = [{"speech": np.zeros(3, dtype=np.float32), "lid_labels": [0]}]
    monkeypatch.setattr(
        stats_module,
        "instantiate",
        lambda config: SimpleNamespace(train=samples, valid=samples),
    )
    config = OmegaConf.create(
        {"dataset": {}, "dataloader": {}, "stats_dir": str(tmp_path)}
    )
    with pytest.raises(ValueError, match="single language string"):
        stats_module.collect_speech_shapes(config)
