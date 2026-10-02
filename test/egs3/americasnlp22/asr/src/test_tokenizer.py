"""Unit tests for the AmericasNLP 2022 tokenizer text helper (no network)."""

from egs3.americasnlp22.asr.src.tokenizer import gather_training_text


def test_gather_training_text_returns_source_raw(tmp_path, corpus_factory) -> None:
    corpus_factory(tmp_path)
    texts = gather_training_text(lang="bzd", recipe_dir=tmp_path, source_dir=tmp_path)
    assert texts == ["raw text 0", "raw text 1"]
