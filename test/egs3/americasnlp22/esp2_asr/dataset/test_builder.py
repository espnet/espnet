"""Unit tests for the AmericasNLP 2022 dataset builder (no network access)."""

import shutil
import tarfile
from types import SimpleNamespace

import pytest

from egs3.americasnlp22.esp2_asr.dataset.builder import (
    AmericasNLP22Builder,
    resolve_language,
    resolve_source_root,
)

_BRIBRI_URL = (
    "https://rcweb.dartmouth.edu/homes/f00458c/americasnlp2/BribriTrainDev.tar.gz"
)


@pytest.fixture
def archive_server(tmp_path, corpus_factory, monkeypatch):
    """Serve a fixture Bribri archive in place of the download, logging URLs."""
    staging = tmp_path / "staging"
    staging.mkdir()
    corpus_factory(staging)
    archive = tmp_path / "BribriTrainDev.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staging / "Bribri", arcname="Bribri")
    calls = []

    def fake_download(url, dst_path, **kwargs):
        calls.append(url)
        shutil.copy(archive, dst_path)

    monkeypatch.setattr(
        "egs3.americasnlp22.esp2_asr.dataset.builder.download_url", fake_download
    )
    monkeypatch.delenv("AMERICASNLP22", raising=False)
    return SimpleNamespace(archive=archive, calls=calls)


def test_resolve_language_maps_codes() -> None:
    assert resolve_language("bzd") == "Bribri"
    assert resolve_language("tav") == "Waikhana"


def test_resolve_language_rejects_unknown_code() -> None:
    try:
        resolve_language("zzz")
    except ValueError as e:
        assert "Unknown language" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_is_source_prepared_detects_prepared_source(tmp_path, corpus_factory) -> None:
    corpus_factory(tmp_path)
    builder = AmericasNLP22Builder()
    assert builder.is_source_prepared(tmp_path, lang="bzd", source_dir=tmp_path)
    # The BaseSystem flow forwards every `create_dataset` block key, including
    # TEMPLATE-inherited extras the builder must tolerate.
    assert builder.is_source_prepared(
        tmp_path,
        lang="bzd",
        source_dir=tmp_path,
        func="src.creating_dataset.create_dataset",
        dataset_dir="/path/to/your/dataset",
    )


def test_is_source_prepared_false_when_split_missing(tmp_path, corpus_factory) -> None:
    corpus_factory(tmp_path)
    (tmp_path / "Bribri" / "dev" / "meta.tsv").unlink()
    builder = AmericasNLP22Builder()
    assert not builder.is_source_prepared(tmp_path, lang="bzd", source_dir=tmp_path)


def test_explicit_source_dir_wins_over_recipe_downloads(
    tmp_path, corpus_factory
) -> None:
    # The recipe-local download directory holds one language and the explicit
    # source_dir another: the requested language must be found in source_dir.
    corpus_factory(tmp_path / "downloads", language="Bribri")
    source_dir = tmp_path / "corpus"
    corpus_factory(source_dir, language="Guarani")
    builder = AmericasNLP22Builder()
    assert resolve_source_root(tmp_path, source_dir=source_dir) == source_dir
    assert builder.is_source_prepared(tmp_path, lang="gn", source_dir=source_dir)
    # Without source_dir, the recipe-local directory is still used.
    assert resolve_source_root(tmp_path) == tmp_path / "downloads"


def test_prepare_source_extracts_local_archive(tmp_path, archive_server) -> None:
    downloads = tmp_path / "downloads"
    builder = AmericasNLP22Builder()
    # No corpus under <recipe_dir>/downloads yet: not prepared, but
    # prepare_source fills it in through the (monkeypatched) download path.
    assert not builder.is_source_prepared(tmp_path, lang="bzd")
    builder.prepare_source(tmp_path, lang="bzd")
    assert archive_server.calls == [_BRIBRI_URL]
    assert (downloads / "Bribri" / "train" / "meta.tsv").is_file()
    assert (downloads / "Bribri" / "dev" / "meta.tsv").is_file()
    # The downloaded archive is cleaned up after extraction.
    assert not (downloads / "BribriTrainDev.tar.gz").exists()
    # Idempotent: the second pass finds the source and downloads nothing.
    builder.prepare_source(tmp_path, lang="bzd")
    assert builder.is_source_prepared(tmp_path, lang="bzd")
    assert builder.is_built(tmp_path, lang="bzd")
    assert len(archive_server.calls) == 1


def test_prepare_source_discards_truncated_archive(tmp_path, archive_server) -> None:
    # An interrupted download leaves a truncated archive behind; extracting it
    # fails, and the archive is removed so that the next run downloads again.
    archive = tmp_path / "downloads" / "BribriTrainDev.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(archive_server.archive.read_bytes()[:100])
    builder = AmericasNLP22Builder()
    with pytest.raises((tarfile.TarError, EOFError)):
        builder.prepare_source(tmp_path, lang="bzd")
    assert archive_server.calls == []
    assert not archive.exists()

    builder.prepare_source(tmp_path, lang="bzd")
    assert archive_server.calls == [_BRIBRI_URL]
    assert builder.is_source_prepared(tmp_path, lang="bzd")


def test_prepare_source_requires_lang(tmp_path) -> None:
    builder = AmericasNLP22Builder()
    try:
        builder.prepare_source(tmp_path)
    except ValueError as e:
        assert "`lang`" in str(e)
    else:
        raise AssertionError("expected ValueError")
