"""Dataset builder checks must recognize complete artifacts without side effects."""

import pytest

from egs3.librimix.esp2_tse.dataset import DatasetBuilder


def test_missing_sources_do_not_create_files(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBRISPEECH", raising=False)
    monkeypatch.delenv("LIBRIMIX", raising=False)
    builder = DatasetBuilder()
    assert not builder.is_source_prepared(recipe_dir=tmp_path)
    assert not builder.is_built(recipe_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(FileNotFoundError, match="LibriSpeech"):
        builder.prepare_source(recipe_dir=tmp_path)


def test_empty_split_directories_are_not_built(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBRIMIX", raising=False)
    root = tmp_path / "data"
    (root / "LibriMix").mkdir(parents=True)
    for split in ("train", "train-100", "train-360", "dev", "test"):
        (root / f"2mix_16k_max_{split}_mix-clean").mkdir()
    assert not DatasetBuilder().is_built(recipe_dir=tmp_path)


def test_manifest_only_tree_is_not_built(tmp_path, monkeypatch):
    from egs3.librimix.esp2_tse.dataset import builder as module

    monkeypatch.delenv("LIBRIMIX", raising=False)
    root = tmp_path / "data"
    (root / "LibriMix").mkdir(parents=True)
    for split in module._CFG["required_splits"]:
        folder = root / split
        folder.mkdir()
        for name in (
            "wav.scp",
            "spk1.scp",
            "spk2.scp",
            "text_spk1",
            "text_spk2",
            "spk2enroll.json" if "train" in split else "mixture2enrollment",
        ):
            (folder / name).touch()
    assert not DatasetBuilder().is_built(recipe_dir=tmp_path)


def test_yaml_noise_path_and_writable_copy_reused(tmp_path, monkeypatch):
    from pathlib import Path

    from egs3.librimix.esp2_tse.dataset import builder as module

    source = tmp_path / "LibriSpeech"
    for split in (
        "train-clean-100",
        "train-clean-360",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ):
        (source / split).mkdir(parents=True)
    external = tmp_path / "external_noise"
    external.mkdir()
    local = tmp_path / "data/wham_noise"
    local.mkdir(parents=True)

    def prepare(wham_noise, **kwargs):
        assert isinstance(wham_noise, Path)
        assert wham_noise == external
        return local

    monkeypatch.setattr(module, "_prepare_wham_noise", prepare)
    monkeypatch.setattr(module, "_prepare_librimix_scripts", lambda **kwargs: None)
    monkeypatch.setattr(module, "download_url", lambda *args, **kwargs: None)
    builder = DatasetBuilder()
    builder.prepare_source(
        recipe_dir=tmp_path, source_dir=source, wham_noise=str(external)
    )

    def build(**kwargs):
        assert kwargs["wham_noise"] == local

    monkeypatch.setattr(module, "_build_librimix", build)
    builder.build(recipe_dir=tmp_path, source_dir=source, wham_noise=str(external))
