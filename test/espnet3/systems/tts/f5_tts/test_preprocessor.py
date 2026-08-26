import importlib.util

import numpy as np
import pytest

from espnet3.systems.tts.f5_tts.preprocessor import F5PinyinPreprocessor

pinyin_deps = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("rjieba", "pypinyin")),
    reason="requires the optional rjieba and pypinyin dependencies",
)


@pytest.fixture
def vocab_file(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text(" \na\nb\nc\n", encoding="utf-8")
    return str(path)


def test_vocab_size_counts_every_line(vocab_file):
    assert F5PinyinPreprocessor(vocab_file).vocab_size == 4


@pinyin_deps
def test_call_replaces_text_with_int64_ids(vocab_file):
    prep = F5PinyinPreprocessor(vocab_file)
    out = prep({"text": "abc"})
    assert isinstance(out["text"], np.ndarray)
    assert out["text"].dtype == np.int64
    assert out["text"].tolist() == [1, 2, 3]


@pinyin_deps
def test_unknown_tokens_fall_back_to_zero(vocab_file):
    """F5's vocab has no <unk>; this is the repo-exact fallback."""
    out = F5PinyinPreprocessor(vocab_file)({"text": "az"})
    assert out["text"].tolist() == [1, 0]


@pinyin_deps
def test_other_keys_pass_through_untouched(vocab_file):
    """The waveform must reach the model unmodified; only text is tokenized."""
    speech = np.zeros(16, dtype=np.float32)
    out = F5PinyinPreprocessor(vocab_file)({"text": "a", "speech": speech})
    assert out["speech"] is speech


@pinyin_deps
def test_text_name_is_configurable(vocab_file):
    prep = F5PinyinPreprocessor(vocab_file, text_name="transcript")
    out = prep({"transcript": "a"})
    assert out["transcript"].tolist() == [1]


def test_train_flag_is_accepted_for_collect_stats(vocab_file):
    """collect_stats toggles train/valid; the flag must not be rejected."""
    assert F5PinyinPreprocessor(vocab_file, train=False).train is False
