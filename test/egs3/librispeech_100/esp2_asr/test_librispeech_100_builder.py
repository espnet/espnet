import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

import egs3.librispeech_100.esp2_asr.dataset.builder as builder_mod
from egs3.librispeech_100.esp2_asr.dataset import Dataset, DatasetBuilder
from espnet3.systems.base.system import BaseSystem

SPLITS = ["train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other"]


def _flac_bytes() -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(160, dtype=np.float32), 16000, format="FLAC")
    return buf.getvalue()


def _add_file(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _write_archive(path: Path, split: str) -> None:
    with tarfile.open(path, "w:gz") as tar:
        _add_file(tar, "LibriSpeech/LICENSE.TXT", b"license\n")
        chapter = f"LibriSpeech/{split}/19/198"
        _add_file(tar, f"{chapter}/19-198.trans.txt", b"19-198-0000 HELLO WORLD\n")
        _add_file(tar, f"{chapter}/19-198-0000.flac", _flac_bytes())


def _write_tree(librispeech_root: Path, splits) -> None:
    for split in splits:
        chapter = librispeech_root / split / "19" / "198"
        chapter.mkdir(parents=True)
        (chapter / "19-198.trans.txt").write_text("19-198-0000 HELLO WORLD\n")
        (chapter / "19-198-0000.flac").write_bytes(_flac_bytes())


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    """Serve small LibriSpeech archives over file:// and count fetches."""
    mirror_dir = tmp_path / "mirror"
    mirror_dir.mkdir()
    for split in SPLITS:
        _write_archive(mirror_dir / f"{split}.tar.gz", split)
    monkeypatch.setitem(builder_mod._CFG, "download_url", mirror_dir.as_uri())

    fetched = []
    real_download_url = builder_mod.download_url

    def counting_download_url(url, dst_path, logger=None, step_percent=5):
        fetched.append(url.rsplit("/", 1)[-1])
        real_download_url(url, dst_path, logger=logger, step_percent=step_percent)

    monkeypatch.setattr(builder_mod, "download_url", counting_download_url)
    return fetched


@pytest.fixture
def offline(monkeypatch):
    def fail_download_url(*args, **kwargs):
        raise AssertionError("network access is not expected")

    monkeypatch.setattr(builder_mod, "download_url", fail_download_url)


@pytest.fixture(autouse=True)
def clear_librispeech_env(monkeypatch):
    monkeypatch.delenv("LIBRISPEECH", raising=False)


@pytest.fixture
def recipe_dir(tmp_path):
    path = tmp_path / "recipe"
    path.mkdir()
    return path


def _run_create_dataset(recipe_dir: Path, **builder_kwargs) -> None:
    config = OmegaConf.create(
        {
            "exp_dir": str(recipe_dir.parent / "exp"),
            "recipe_dir": str(recipe_dir),
            "create_dataset": {"recipe_dir": str(recipe_dir), **builder_kwargs},
            "dataset": {"train": [{"data_src": "librispeech_100/esp2_asr"}]},
        }
    )
    BaseSystem(training_config=config).create_dataset()


def test_create_dataset_downloads_and_extracts_all_splits(recipe_dir, mirror):
    _run_create_dataset(recipe_dir)

    download_dir = recipe_dir / "download"
    assert sorted(mirror) == sorted(f"{s}.tar.gz" for s in SPLITS)
    for split in SPLITS:
        assert (download_dir / "LibriSpeech" / split).is_dir()
        assert (download_dir / f"{split}.tar.gz").is_file()
    assert (download_dir / "LibriSpeech" / "LICENSE.TXT").is_file()
    assert not list(download_dir.glob("*.part"))
    assert not list(download_dir.glob(".*.extracting"))
    assert DatasetBuilder().is_source_prepared(recipe_dir=recipe_dir)

    dataset = Dataset(split="dev-clean", recipe_dir=recipe_dir)
    assert len(dataset) == 1
    assert dataset[0]["text"] == "HELLO WORLD"

    _run_create_dataset(recipe_dir)
    assert len(mirror) == len(SPLITS)


@pytest.mark.parametrize("layout", ["parent", "root"])
def test_create_dataset_reuses_corpus_from_env_without_network(
    tmp_path, recipe_dir, offline, monkeypatch, layout
):
    corpus = tmp_path / "shared"
    _write_tree(corpus / "LibriSpeech", SPLITS)
    env_path = corpus if layout == "parent" else corpus / "LibriSpeech"
    monkeypatch.setenv("LIBRISPEECH", str(env_path))

    _run_create_dataset(recipe_dir)

    assert not (recipe_dir / "download").exists()
    dataset = Dataset(split="test-other", recipe_dir=recipe_dir)
    assert dataset.librispeech_root == corpus / "LibriSpeech"


def test_create_dataset_reuses_corpus_from_source_dir_without_network(
    tmp_path, recipe_dir, offline
):
    corpus = tmp_path / "shared"
    _write_tree(corpus / "LibriSpeech", SPLITS)

    _run_create_dataset(recipe_dir, source_dir=str(corpus))

    assert not (recipe_dir / "download").exists()


def test_dataset_path_takes_precedence_over_env(
    tmp_path, recipe_dir, offline, monkeypatch
):
    _write_tree(recipe_dir / "download" / "LibriSpeech", SPLITS)
    other = tmp_path / "other"
    _write_tree(other / "LibriSpeech", SPLITS)
    monkeypatch.setenv("LIBRISPEECH", str(other))

    _run_create_dataset(recipe_dir)

    dataset = Dataset(split="dev-clean", recipe_dir=recipe_dir)
    assert dataset.librispeech_root == recipe_dir / "download" / "LibriSpeech"


def test_create_dataset_completes_partial_download_dir(recipe_dir, mirror):
    download_dir = recipe_dir / "download"
    _write_tree(download_dir / "LibriSpeech", SPLITS[:3])
    marker = download_dir / "LibriSpeech" / "dev-clean" / "keep.txt"
    marker.write_text("user data")
    _write_archive(download_dir / "test-clean.tar.gz", "test-clean")

    _run_create_dataset(recipe_dir)

    assert mirror == ["test-other.tar.gz"]
    assert marker.read_text() == "user data"
    assert DatasetBuilder().is_source_prepared(recipe_dir=recipe_dir)


def test_create_dataset_rejects_partial_external_corpus(
    tmp_path, recipe_dir, mirror, monkeypatch
):
    corpus = tmp_path / "shared"
    _write_tree(corpus / "LibriSpeech", SPLITS[:4])
    monkeypatch.setenv("LIBRISPEECH", str(corpus))

    with pytest.raises(
        FileNotFoundError, match="missing split directories: test-other"
    ):
        _run_create_dataset(recipe_dir)

    assert mirror == []
    assert not (corpus / "LibriSpeech" / "test-other").exists()
    assert not (recipe_dir / "download").exists()


def test_create_dataset_reports_failed_download(recipe_dir, tmp_path, monkeypatch):
    monkeypatch.setitem(
        builder_mod._CFG, "download_url", (tmp_path / "missing").as_uri()
    )

    with pytest.raises(RuntimeError, match="(?s)Failed to download .*LIBRISPEECH"):
        _run_create_dataset(recipe_dir)

    download_dir = recipe_dir / "download"
    assert not list(download_dir.glob("*.part"))
    assert not DatasetBuilder().is_source_prepared(recipe_dir=recipe_dir)


def test_create_dataset_reports_corrupt_archive(recipe_dir, offline):
    download_dir = recipe_dir / "download"
    _write_tree(download_dir / "LibriSpeech", SPLITS[1:])
    (download_dir / "train-clean-100.tar.gz").write_bytes(b"not a tarball")

    with pytest.raises(
        RuntimeError, match="(?s)Failed to extract .*Delete it and rerun"
    ):
        _run_create_dataset(recipe_dir)

    assert not (download_dir / "LibriSpeech" / "train-clean-100").exists()
    assert not list(download_dir.glob(".*.extracting"))


def test_readiness_checks_have_no_side_effects(recipe_dir, offline):
    builder = DatasetBuilder()

    assert not builder.is_source_prepared(recipe_dir=recipe_dir)
    assert not builder.is_built(recipe_dir=recipe_dir)
    assert list(recipe_dir.iterdir()) == []


def test_dataset_without_source_asks_for_create_dataset(recipe_dir, offline):
    with pytest.raises(FileNotFoundError, match="create_dataset"):
        Dataset(split="dev-clean", recipe_dir=recipe_dir)

    assert list(recipe_dir.iterdir()) == []
