"""Tests for espnet2/speechlm/model/speechlm/lm/parallel.py.

Tests ParallelHFModel factory, build_parallel_hf_class, and ParallelLLM
with mock HuggingFace architectures. Uses the transformers stub from conftest.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from espnet2.speechlm.model.speechlm.multimodal_io.abs_io import AbsIO


# ---------------------------------------------------------------------------
# Mock IOs for parallel model tests
# ---------------------------------------------------------------------------
class _MockDiscreteIO(AbsIO):
    """Minimal discrete IO for building a parallel model."""

    def __init__(self, vocab_size=100, n_stream=1, modality="text"):
        super().__init__(modality=modality, is_discrete=True)
        self._vocab_size = vocab_size
        self._n_stream = n_stream

    def num_stream(self):
        return self._n_stream

    def get_vocabulary(self):
        return [f"{self.modality}_tok_{i}" for i in range(self._vocab_size)]

    def get_stream_interval(self):
        per_stream = self._vocab_size // self._n_stream
        return [(i * per_stream, (i + 1) * per_stream) for i in range(self._n_stream)]

    def preprocess(self, data):
        return (
            np.zeros((3, self._n_stream), dtype=np.int64),
            None,
            np.ones((3, self._n_stream), dtype=np.float32),
        )

    def find_length(self, data):
        return 3

    def copy_for_worker(self):
        return self

    def feature_dim(self):
        return None

    def encode_batch(self, feats, lengths):
        return torch.zeros(feats.shape[0], 3, self._n_stream, dtype=torch.long)

    def decode_batch(self, codes, lengths):
        return [None] * codes.shape[0]

    def dummy_forward(self, ref_tensor=None):
        return torch.zeros(1, requires_grad=True)


class _MockContinuousIO(AbsIO):
    """Minimal continuous IO for building a parallel model."""

    def __init__(self, feat_dim=80):
        super().__init__(modality="audio", is_discrete=False)
        self._feat_dim = feat_dim

    def num_stream(self):
        return None

    def get_vocabulary(self):
        return None

    def get_stream_interval(self):
        return None

    def feature_dim(self):
        return self._feat_dim

    def encode_batch(self, feats, lengths):
        return feats

    def decode_batch(self, codes, lengths):
        return [None] * codes.shape[0]

    def copy_for_worker(self):
        return self

    def dummy_forward(self, ref_tensor=None):
        return torch.zeros(1, requires_grad=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _build_text_io():
    return _MockDiscreteIO(vocab_size=100, n_stream=1, modality="text")


def _build_audio_io():
    return _MockDiscreteIO(vocab_size=400, n_stream=4, modality="audio")


def _build_continuous_io():
    return _MockContinuousIO(feat_dim=80)


def _make_multimodal_io(include_continuous=False):
    ios = nn.ModuleDict(
        {
            "text": _build_text_io(),
            "discrete_audio": _build_audio_io(),
        }
    )
    if include_continuous:
        ios["continuous_audio"] = _build_continuous_io()
    return ios


def _make_vocab_and_intervals(text_io, audio_io):
    """Build vocab and intervals matching how SpeechLMJobTemplate does it."""
    # Special tokens
    vocab = [
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|eot|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|text|>",
        "<|audio|>",
        "<|image|>",
        "<|video|>",
        "<|toolcall|>",
    ]
    while len(vocab) < 256:
        vocab.append(f"<|unused_{len(vocab)}|>")

    intervals = {"special_token": [(0, 256)]}

    # Text IO
    start = 256
    vocab.extend(text_io.get_vocabulary())
    intervals["text"] = [
        (start + s, start + e) for s, e in text_io.get_stream_interval()
    ]
    start = len(vocab)

    # Audio IO
    vocab.extend(audio_io.get_vocabulary())
    intervals["discrete_audio"] = [
        (start + s, start + e) for s, e in audio_io.get_stream_interval()
    ]

    return vocab, intervals


@pytest.fixture
def model_components():
    """Create model components for parallel LLM tests."""
    text_io = _build_text_io()
    audio_io = _build_audio_io()
    multimodal_io = nn.ModuleDict({"text": text_io, "discrete_audio": audio_io})
    vocab, intervals = _make_vocab_and_intervals(text_io, audio_io)
    return multimodal_io, vocab, intervals


@pytest.fixture
def parallel_model(model_components):
    """Build a ParallelLLM model using mock architecture."""
    from espnet2.speechlm.model.speechlm.lm.parallel import build_parallel_hf_class

    multimodal_io, vocab, intervals = model_components
    cls = build_parallel_hf_class("mock-model")
    model = cls.from_pretrained(
        "mock-model",
        multimodal_io=multimodal_io,
        vocab=vocab,
        vocab_intervals=intervals,
    )
    return model


# ---------------------------------------------------------------------------
# build_parallel_hf_class
# ---------------------------------------------------------------------------
class TestBuildParallelHFClass:
    def test_returns_class(self):
        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )

        cls = build_parallel_hf_class("mock-model")
        assert isinstance(cls, type)

    def test_parallel_llm_is_subclass(self):
        import transformers

        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )

        cls = build_parallel_hf_class("mock-model")
        assert issubclass(cls, transformers.MockModel)


# ---------------------------------------------------------------------------
# from_pretrained
# ---------------------------------------------------------------------------
class TestFromPretrained:
    def test_rebuilds_embeddings(self, parallel_model, model_components):
        _, vocab, intervals = model_components
        expected_vocab_size = max(end for ivs in intervals.values() for _, end in ivs)
        assert parallel_model.model.embed_tokens.weight.shape[0] == expected_vocab_size
        assert parallel_model.lm_head.weight.shape[0] == expected_vocab_size

    def test_stream_emb(self, parallel_model):
        assert hasattr(parallel_model, "stream_emb")
        assert parallel_model.stream_emb.weight.shape[0] == parallel_model.num_stream
        assert (
            parallel_model.stream_emb.weight.shape[1]
            == parallel_model.config.hidden_size
        )

    def test_adaptor_for_continuous(self):
        """Continuous IOs should get linear adaptors."""
        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )

        text_io = _build_text_io()
        audio_io = _build_audio_io()
        cont_io = _build_continuous_io()
        multimodal_io = nn.ModuleDict(
            {"text": text_io, "discrete_audio": audio_io, "continuous_audio": cont_io}
        )
        vocab, intervals = _make_vocab_and_intervals(text_io, audio_io)
        cls = build_parallel_hf_class("mock-model")
        model = cls.from_pretrained(
            "mock-model",
            multimodal_io=multimodal_io,
            vocab=vocab,
            vocab_intervals=intervals,
        )
        assert "continuous_audio" in model.adaptor
        assert isinstance(model.adaptor["continuous_audio"], nn.Linear)
        assert model.adaptor["continuous_audio"].in_features == 80

    def test_loss_intervals(self, parallel_model, model_components):
        _, _, intervals = model_components
        # loss_intervals should cover discrete_audio but not text or special_token
        assert len(parallel_model.loss_intervals) > 0
        for start, end in parallel_model.loss_intervals:
            assert start < end
            # Should not include special_token or text intervals
            assert start >= intervals["text"][0][1]

    def test_no_discrete_raises(self):
        """All continuous IOs should raise ValueError."""
        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )

        cont_io = _build_continuous_io()
        multimodal_io = nn.ModuleDict({"continuous_audio": cont_io})
        vocab = ["<|pad|>"] + [f"v{i}" for i in range(255)]
        intervals = {"special_token": [(0, 256)]}
        cls = build_parallel_hf_class("mock-model")
        with pytest.raises(ValueError, match="all IOs being continuous"):
            cls.from_pretrained(
                "mock-model",
                multimodal_io=multimodal_io,
                vocab=vocab,
                vocab_intervals=intervals,
            )


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------
class TestForward:
    def test_returns_loss_and_stats(self, parallel_model):
        batch_size, seq_len = 2, 10
        num_stream = parallel_model.num_stream
        seqs = torch.zeros(batch_size, seq_len, num_stream, dtype=torch.long)
        loss_masks = torch.ones(batch_size, seq_len, num_stream)

        parallel_model.train()
        out = parallel_model(seqs=seqs, loss_masks=loss_masks)
        assert "loss" in out
        assert "stats" in out

    def test_loss_shape(self, parallel_model):
        batch_size, seq_len = 2, 10
        num_stream = parallel_model.num_stream
        seqs = torch.zeros(batch_size, seq_len, num_stream, dtype=torch.long)
        loss_masks = torch.ones(batch_size, seq_len, num_stream)

        parallel_model.train()
        out = parallel_model(seqs=seqs, loss_masks=loss_masks)
        assert out["loss"].ndim == 0  # scalar


# ---------------------------------------------------------------------------
# _embed
# ---------------------------------------------------------------------------
class TestEmbed:
    def test_embed_discrete_features(self, parallel_model):
        batch_size, seq_len = 1, 5
        num_stream = parallel_model.num_stream
        input_ids = torch.zeros(batch_size, seq_len, num_stream, dtype=torch.long)
        kwargs = {"seqs": input_ids}
        embeds = parallel_model._embed(input_ids, kwargs)
        assert embeds.shape == (batch_size, seq_len, parallel_model.config.hidden_size)

    def test_embed_continuous_features(self):
        """Continuous features should be projected through adaptor."""
        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )

        text_io = _build_text_io()
        audio_io = _build_audio_io()
        cont_io = _build_continuous_io()
        multimodal_io = nn.ModuleDict(
            {"text": text_io, "discrete_audio": audio_io, "continuous_audio": cont_io}
        )
        vocab, intervals = _make_vocab_and_intervals(text_io, audio_io)
        cls = build_parallel_hf_class("mock-model")
        model = cls.from_pretrained(
            "mock-model",
            multimodal_io=multimodal_io,
            vocab=vocab,
            vocab_intervals=intervals,
        )

        batch_size, seq_len = 1, 10
        num_stream = model.num_stream
        input_ids = torch.zeros(batch_size, seq_len, num_stream, dtype=torch.long)
        kwargs = {
            "seqs": input_ids,
            "continuous_audio_indices": torch.tensor([[0, 2, 3]]),
            "continuous_audio_feats": torch.randn(1, 3, 80),
            "continuous_audio_lengths": torch.tensor([3]),
        }
        embeds = model._embed(input_ids, kwargs)
        assert embeds.shape == (batch_size, seq_len, model.config.hidden_size)


# ---------------------------------------------------------------------------
# _loss
# ---------------------------------------------------------------------------
class TestLoss:
    def test_loss_next_token_prediction(self, parallel_model):
        """Hidden states should be shifted by 1 for next-token prediction."""
        batch_size, seq_len = 1, 6
        num_stream = parallel_model.num_stream
        hidden_dim = parallel_model.config.hidden_size

        hidden_states = torch.randn(batch_size, seq_len, num_stream, hidden_dim)
        input_ids = torch.zeros(batch_size, seq_len, num_stream, dtype=torch.long)
        loss_mask = torch.ones(batch_size, seq_len, num_stream)

        parallel_model.train()
        loss, stats = parallel_model._loss(
            hidden_states=hidden_states,
            input_ids=input_ids,
            loss_mask=loss_mask,
            router_logits=None,
        )
        assert loss.ndim == 0  # scalar loss

    def test_loss_interval_computation(self, parallel_model):
        """Per-stream interval-based loss should be computed."""
        assert len(parallel_model.loss_intervals) > 0
        batch_size, seq_len = 1, 6
        num_stream = parallel_model.num_stream
        hidden_dim = parallel_model.config.hidden_size

        hidden_states = torch.randn(batch_size, seq_len, num_stream, hidden_dim)
        # Use tokens in the audio interval
        start, end = parallel_model.loss_intervals[0]
        input_ids = torch.full(
            (batch_size, seq_len, num_stream), start, dtype=torch.long
        )
        loss_mask = torch.ones(batch_size, seq_len, num_stream)

        parallel_model.train()
        loss, stats = parallel_model._loss(
            hidden_states=hidden_states,
            input_ids=input_ids,
            loss_mask=loss_mask,
            router_logits=None,
        )
        assert "loss" in stats


# ---------------------------------------------------------------------------
# prepare_inference
# ---------------------------------------------------------------------------
class TestPrepareInference:
    def test_registers_buffers(self, parallel_model):
        parallel_model.prepare_inference()
        assert hasattr(parallel_model, "assistant_token")
        assert hasattr(parallel_model, "audio_token")
        assert hasattr(parallel_model, "text_token")
        assert hasattr(parallel_model, "eos_token")
        assert hasattr(parallel_model, "eot_token")
        assert hasattr(parallel_model, "modality_mask")
        assert parallel_model.assistant_token.shape == (1, 1, parallel_model.num_stream)


# ---------------------------------------------------------------------------
# _logits_to_token
# ---------------------------------------------------------------------------
class TestLogitsToToken:
    def test_greedy(self, parallel_model):
        logits = torch.randn(1, 1, parallel_model.num_stream, len(parallel_model.vocab))
        tokens = parallel_model._logits_to_token(logits, temperature=0, topk=1)
        expected = logits.argmax(-1)
        assert torch.equal(tokens, expected)

    def test_sampling(self, parallel_model):
        torch.manual_seed(42)
        logits = torch.randn(1, 1, parallel_model.num_stream, len(parallel_model.vocab))
        tokens = parallel_model._logits_to_token(logits, temperature=1.0, topk=10)
        assert tokens.shape == (1, 1, parallel_model.num_stream)
        # Tokens should be valid indices
        assert (tokens >= 0).all()
        assert (tokens < len(parallel_model.vocab)).all()


# ---------------------------------------------------------------------------
# train_dtype (static method-like behavior via ds_config)
# ---------------------------------------------------------------------------
class TestTrainDtype:
    def test_bf16(self, parallel_model):
        # train_dtype is on the trainer, but we test the logic directly
        from espnet2.speechlm.trainer.deepspeed_trainer import DeepSpeedTrainer

        ds_config = {"bf16": {"enabled": True}}
        assert DeepSpeedTrainer.train_dtype(None, ds_config) == torch.bfloat16

    def test_fp16(self, parallel_model):
        from espnet2.speechlm.trainer.deepspeed_trainer import DeepSpeedTrainer

        ds_config = {"fp16": {"enabled": True}}
        assert DeepSpeedTrainer.train_dtype(None, ds_config) == torch.float16

    def test_default(self, parallel_model):
        from espnet2.speechlm.trainer.deepspeed_trainer import DeepSpeedTrainer

        ds_config = {}
        assert DeepSpeedTrainer.train_dtype(None, ds_config) == torch.float
