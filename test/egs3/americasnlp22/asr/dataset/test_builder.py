"""Unit tests for the AmericasNLP 2022 dataset builder (no network access)."""

import shutil
import tarfile

from egs3.americasnlp22.asr.dataset.builder import (
    AmericasNLP22Builder,
    resolve_language,
)


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


def test_prepare_source_extracts_local_archive(
    tmp_path, corpus_factory, monkeypatch
) -> None:
    # Stage the corpus and bundle it the way the shared-task archive looks.
    staging = tmp_path / "staging"
    staging.mkdir()
    corpus_factory(staging)
    fixture_tar = tmp_path / "BribriTrainDev.tar.gz"
    with tarfile.open(fixture_tar, "w:gz") as tar:
        tar.add(staging / "Bribri", arcname="Bribri")

    downloads = tmp_path / "downloads"
    calls = []

    def fake_download(url, dst_path, **kwargs):
        calls.append(url)
        shutil.copy(fixture_tar, dst_path)

    monkeypatch.setattr(
        "egs3.americasnlp22.asr.dataset.builder.download_url", fake_download
    )

    builder = AmericasNLP22Builder()
    # No corpus under <recipe_dir>/downloads yet: not prepared, but
    # prepare_source fills it in through the (monkeypatched) download path.
    assert not builder.is_source_prepared(tmp_path, lang="bzd")
    builder.prepare_source(tmp_path, lang="bzd")
    assert calls == [
        "https://rcweb.dartmouth.edu/homes/f00458c/americasnlp2/BribriTrainDev.tar.gz"
    ]
    assert (downloads / "Bribri" / "train" / "meta.tsv").is_file()
    assert (downloads / "Bribri" / "dev" / "meta.tsv").is_file()
    # The downloaded archive is cleaned up after extraction.
    assert not (downloads / "BribriTrainDev.tar.gz").exists()
    # Idempotent: the second pass finds the source and downloads nothing.
    builder.prepare_source(tmp_path, lang="bzd")
    assert builder.is_source_prepared(tmp_path, lang="bzd")
    assert builder.is_built(tmp_path, lang="bzd")
    assert len(calls) == 1


def test_prepare_source_requires_lang(tmp_path) -> None:
    builder = AmericasNLP22Builder()
    try:
        builder.prepare_source(tmp_path)
    except ValueError as e:
        assert "`lang`" in str(e)
    else:
        raise AssertionError("expected ValueError")
