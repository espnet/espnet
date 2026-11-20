"""Unit tests for the AmericasNLP 2022 dataset (no network access)."""

import numpy as np

from egs3.americasnlp22.esp2_asr.dataset.dataset import AmericasNLP22Dataset
from espnet3.components.data.base_dataset import BaseDataset


def test_dataset_declares_contract_fields() -> None:
    assert [(f.name, f.kind) for f in AmericasNLP22Dataset.fields] == [
        ("speech", "audio"),
        ("text", "text"),
    ]


def test_dataset_returns_only_preprocessor_fields(tmp_path, corpus_factory) -> None:
    corpus_factory(tmp_path)
    dataset = AmericasNLP22Dataset(
        lang="bzd", split="train", recipe_dir=tmp_path, source_dir=tmp_path
    )
    assert isinstance(dataset, BaseDataset)
    assert len(dataset) == 2

    # The first item also passes BaseDataset's one-time `fields` check.
    sample = dataset[0]
    assert sorted(sample.keys()) == ["speech", "text"]
    assert sample["text"] == "raw text 0"
    assert sample["speech"].dtype == np.float32
    # Corpus order: rows follow the sorted corpus utterance ids.
    assert dataset[1]["text"] == "raw text 1"


def test_dataset_max_wav_duration_filters_all(tmp_path, corpus_factory) -> None:
    corpus_factory(tmp_path)
    # All synthetic utts are short; a 0.4 s cap filters all of them, which
    # surfaces as the "no usable utterances" error with a helpful message.
    try:
        AmericasNLP22Dataset(
            lang="bzd",
            split="train",
            recipe_dir=tmp_path,
            source_dir=tmp_path,
            max_wav_duration=0.4,
        )
    except RuntimeError as e:
        assert "No usable utterances" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_dataset_min_wav_duration_drops_short_utts(tmp_path, corpus_factory) -> None:
    corpus_factory(tmp_path)
    # The synthetic train utts last 0.5 s and 0.6 s; a 0.55 s floor keeps
    # only the second one.
    dataset = AmericasNLP22Dataset(
        lang="bzd",
        split="train",
        recipe_dir=tmp_path,
        source_dir=tmp_path,
        min_wav_duration=0.55,
    )
    assert len(dataset) == 1
    assert dataset[0]["text"] == "raw text 1"


def test_dataset_max_wav_duration_keeps_short_utts(tmp_path, corpus_factory) -> None:
    corpus_factory(tmp_path)
    dataset = AmericasNLP22Dataset(
        lang="bzd",
        split="train",
        recipe_dir=tmp_path,
        source_dir=tmp_path,
        max_wav_duration=38,
    )
    assert len(dataset) == 2


def test_dataset_rejects_unknown_split(tmp_path, corpus_factory) -> None:
    corpus_factory(tmp_path)
    try:
        AmericasNLP22Dataset(
            lang="bzd", split="test", recipe_dir=tmp_path, source_dir=tmp_path
        )
    except ValueError as e:
        assert "Unknown split" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_dataset_rejects_unknown_lang(tmp_path, corpus_factory) -> None:
    corpus_factory(tmp_path)
    try:
        AmericasNLP22Dataset(
            lang="zzz", split="train", recipe_dir=tmp_path, source_dir=tmp_path
        )
    except ValueError as e:
        assert "Unknown language" in str(e)
    else:
        raise AssertionError("expected ValueError")
