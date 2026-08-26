import importlib.util

import numpy as np
import pytest

from espnet3.systems.tts.f5_tts.pinyin import (
    build_pinyin_vocab,
    convert_char_to_pinyin,
    f5_pinyin_g2p,
    load_vocab_char_map,
    text_to_pinyin_ids,
)

# `convert_char_to_pinyin` runs rjieba on every input, English included,
# so anything reaching it needs both optional dependencies.
pinyin_deps = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("rjieba", "pypinyin")),
    reason="requires the optional rjieba and pypinyin dependencies",
)


# --- dependency-free helpers -------------------------------------------------


def test_load_vocab_char_map_uses_line_number_as_index(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text(" \na\nb\n", encoding="utf-8")
    vocab = load_vocab_char_map(str(path))
    assert vocab[" "] == 0
    assert vocab["a"] == 1
    assert vocab["b"] == 2


def test_load_vocab_char_map_keeps_a_literal_space_token(tmp_path):
    """F5's vocab.txt has a real space at index 0; stripping it would break ids."""
    path = tmp_path / "vocab.txt"
    path.write_text(" \nx\n", encoding="utf-8")
    assert " " in load_vocab_char_map(str(path))


# --- tokenizer ---------------------------------------------------------------


@pinyin_deps
def test_english_stays_character_level():
    """Emilia_ZH_EN_pinyin is not "pinyin for English"; English stays chars."""
    tokens = convert_char_to_pinyin(["abc"])[0]
    assert tokens == ["a", "b", "c"]


@pinyin_deps
def test_chinese_becomes_toned_pinyin_syllables():
    tokens = convert_char_to_pinyin(["中文"])[0]
    joined = "".join(tokens)
    assert any(c.isdigit() for c in joined), "expected Style.TONE3 tone digits"
    assert not any("一" <= c <= "鿿" for c in joined)


@pinyin_deps
def test_custom_translation_maps_oov_punctuation():
    """Semicolons and curly quotes are folded to ASCII to avoid OOV tokens."""
    tokens = convert_char_to_pinyin(["a;b"])[0]
    assert ";" not in tokens
    assert "," in tokens


@pinyin_deps
def test_g2p_wrapper_takes_one_string_and_returns_one_list():
    assert f5_pinyin_g2p("abc") == convert_char_to_pinyin(["abc"])[0]


@pinyin_deps
def test_batch_entries_are_tokenized_independently():
    batch = convert_char_to_pinyin(["ab", "cd"])
    assert len(batch) == 2
    assert batch[0] == ["a", "b"]


# --- id mapping --------------------------------------------------------------


@pinyin_deps
def test_unknown_tokens_map_to_zero(tmp_path):
    """F5's vocab has no <unk>; the documented fallback is index 0."""
    path = tmp_path / "vocab.txt"
    path.write_text(" \na\n", encoding="utf-8")
    vocab = load_vocab_char_map(str(path))
    ids = text_to_pinyin_ids("az", vocab)
    assert ids.dtype == np.int64
    assert ids[0] == vocab["a"]
    assert ids[1] == 0


# --- vocab construction ------------------------------------------------------


@pinyin_deps
def test_vocab_is_sorted_and_deduplicated():
    vocab = build_pinyin_vocab(["ba", "ab"])
    assert vocab == sorted(set(vocab))


@pinyin_deps
def test_vocab_puts_a_literal_space_at_index_zero():
    """F5's convention is vocab[" "] == 0, achieved by codepoint sorting."""
    vocab = build_pinyin_vocab(["hello world"])
    assert vocab[0] == " "


@pinyin_deps
def test_add_ascii_latin_broadens_coverage():
    plain = build_pinyin_vocab(["ab"])
    seeded = build_pinyin_vocab(["ab"], add_ascii_latin=True)
    assert set(plain).issubset(set(seeded))
    assert "~" in seeded and "~" not in plain


@pinyin_deps
def test_vocab_has_no_espnet_special_symbols():
    """This builds F5's vocab.txt, which carries no <blank>/<unk>/<sos/eos>."""
    vocab = build_pinyin_vocab(["hello"])
    assert not any(t.startswith("<") for t in vocab)


@pinyin_deps
def test_registering_the_g2p_is_idempotent():
    import espnet2.text.phoneme_tokenizer as pt
    from espnet3.systems.tts.f5_tts.pinyin import register_f5_pinyin_g2p

    register_f5_pinyin_g2p()
    register_f5_pinyin_g2p()
    assert pt.g2p_choices.count("f5_pinyin") == 1

    tokenizer = pt.PhonemeTokenizer(g2p_type="f5_pinyin")
    assert tokenizer.text2tokens("abc") == ["a", "b", "c"]
