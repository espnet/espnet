"""Tests for HuggingFaceTextIO."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from espnet2.speechlm.model.speechlm.multimodal_io.text import HuggingFaceTextIO


class _MockTokenizer:
    """Lightweight tokenizer mock that works with or without real transformers."""

    def encode(self, text, **kwargs):
        return list(range(min(len(text.split()) + 2, 50)))

    def decode(self, token_ids, **kwargs):
        return " ".join(f"w{i}" for i in token_ids)

    def get_vocab(self):
        return {f"tok_{i}": i for i in range(80)}


class _MockConfig:
    vocab_size = 100


@pytest.fixture
def text_io():
    with (
        patch(
            "espnet2.speechlm.model.speechlm.multimodal_io.text.AutoTokenizer"
        ) as mock_at,
        patch(
            "espnet2.speechlm.model.speechlm.multimodal_io.text.AutoConfig"
        ) as mock_ac,
    ):
        mock_at.from_pretrained.return_value = _MockTokenizer()
        mock_ac.from_pretrained.return_value = _MockConfig()
        io = HuggingFaceTextIO(tokenizer_name="mock-tokenizer")
    return io


class TestHuggingFaceTextIO:
    def test_init_attributes(self, text_io):
        assert text_io.modality == "text"
        assert text_io.is_discrete is True
        assert text_io.vocab_size == 100

    def test_preprocess_returns_correct_types(self, text_io):
        seq, conti_feat, loss_mask = text_io.preprocess("hello world test")
        assert isinstance(seq, np.ndarray)
        assert seq.dtype == np.int32
        assert seq.ndim == 2
        assert seq.shape[1] == 1
        assert conti_feat is None
        assert isinstance(loss_mask, np.ndarray)
        assert loss_mask.dtype == np.float32
        assert loss_mask.shape == seq.shape

    def test_preprocess_shape_matches_find_length(self, text_io):
        text = "hello world test"
        length = text_io.find_length(text)
        seq, _, _ = text_io.preprocess(text)
        assert seq.shape[0] == length

    def test_find_length_returns_int(self, text_io):
        length = text_io.find_length("some text")
        assert isinstance(length, int)
        assert length > 0

    def test_decode_batch(self, text_io):
        tokens_1stream = torch.tensor([[[1], [2], [3]], [[4], [5], [0]]])
        lengths = torch.tensor([3, 2])
        result = text_io.decode_batch(tokens_1stream, lengths)
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], str)

    def test_num_stream(self, text_io):
        assert text_io.num_stream() == 1

    def test_get_vocabulary(self, text_io):
        vocab = text_io.get_vocabulary()
        assert isinstance(vocab, list)
        assert len(vocab) == text_io.vocab_size
        # Real vocab has 80 tokens, padded to 100
        assert vocab[-1] == "<unused_99>"

    def test_get_stream_interval(self, text_io):
        intervals = text_io.get_stream_interval()
        assert intervals == [(0, 100)]

    def test_get_stream_weight(self, text_io):
        weights = text_io.get_stream_weight()
        assert weights == [1.0]

    def test_copy_for_worker(self, text_io):
        with (
            patch(
                "espnet2.speechlm.model.speechlm.multimodal_io.text.AutoTokenizer"
            ) as mock_at,
            patch(
                "espnet2.speechlm.model.speechlm.multimodal_io.text.AutoConfig"
            ) as mock_ac,
        ):
            mock_at.from_pretrained.return_value = _MockTokenizer()
            mock_ac.from_pretrained.return_value = _MockConfig()
            copy = text_io.copy_for_worker()
        assert isinstance(copy, HuggingFaceTextIO)
        assert copy is not text_io
        assert copy.tokenizer_name == text_io.tokenizer_name

    def test_loss_mask_all_ones(self, text_io):
        _, _, loss_mask = text_io.preprocess("test sentence")
        assert np.all(loss_mask == 1.0)
