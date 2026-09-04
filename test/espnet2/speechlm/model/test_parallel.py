"""Tests for espnet2/speechlm/model/speechlm/lm/parallel.py.

Tests ``ParallelHFModel`` / ``build_parallel_hf_class`` / ``ParallelLLM``
with mock HuggingFace architectures. Patches transformers so tests work
both with the conftest stub and with real transformers installed (CI).

The real Liger kernel in ``lm/loss.py`` requires CUDA, so we
monkeypatch ``LigerFusedLinearCrossEntropyFunction`` with a CPU stub
that returns zero tensors and records call args. The whole file is
skipped (via the sibling ``conftest.py``'s ``collect_ignore_glob``)
when ``liger_kernel.ops.fused_linear_cross_entropy`` is not importable.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from espnet2.speechlm.model.speechlm.multimodal_io.abs_io import AbsIO


# ---------------------------------------------------------------------------
# Mock HF model components (used to patch transformers in CI)
# ---------------------------------------------------------------------------
class _MockConfig:
    architectures = ["MockModel"]
    vocab_size = 100
    hidden_size = 64
    _attn_implementation = "flash_attention_2"


class _MockInnerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, inputs_embeds=None, position_ids=None, **kwargs):
        class _Out:
            pass

        out = _Out()
        out.last_hidden_state = inputs_embeds
        out.past_key_values = None
        out.router_logits = None

        def get(key, default=None):
            return default

        out.get = get
        return out


class _MockHFModel(nn.Module):
    config_class = _MockConfig

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = _MockConfig()
        self.config = config
        self.model = _MockInnerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        config = _MockConfig()
        return cls(config)


# ---------------------------------------------------------------------------
# Fake Liger kernel: records call args, returns zero tensors so callers can
# continue the flow on CPU. Real Liger kernel needs CUDA.
# ---------------------------------------------------------------------------
class _FakeLigerFn:
    calls = []

    @staticmethod
    def apply(*args):
        _FakeLigerFn.calls.append(args)
        return (
            torch.zeros((), requires_grad=True),
            torch.zeros(()),
            torch.zeros(()),
        )


@pytest.fixture(autouse=True)
def _patch_transformers_and_liger(monkeypatch):
    """Patch transformers + Liger kernel for CPU tests.

    Uses pytest's ``monkeypatch`` for every mutation so the cleanup is
    guaranteed (including on test failure). We intentionally do NOT
    use ``patch.object`` + ``with`` here: bare attribute mutations on
    the real ``transformers`` module have leaked into unrelated tests
    in CI; routing everything through ``monkeypatch`` makes the scope
    strictly per-test.
    """
    import transformers

    # Register a fresh mock architecture class under `transformers.MockModel`
    # so `build_parallel_hf_class("mock-model")` can find it via getattr.
    monkeypatch.setattr(transformers, "MockModel", _MockHFModel, raising=False)

    # Return a fresh `_MockConfig` per call — don't share one instance with
    # every caller, and don't leak it into a long-lived mock's return_value.
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *a, **kw: _MockConfig()),
    )

    # Real Liger kernel runs on GPU only; swap in a deterministic CPU stub.
    monkeypatch.setattr(
        "espnet2.speechlm.model.speechlm.lm.loss."
        "LigerFusedLinearCrossEntropyFunction",
        _FakeLigerFn,
    )
    _FakeLigerFn.calls.clear()
    yield


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

    def get_stream_weight(self):
        return [1.0] * self._n_stream

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
        # Shape matches feat_dim so the adaptor linear can consume it during
        # the "dummy graph" pass inside _embed.
        return torch.zeros(1, self._feat_dim, requires_grad=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _build_text_io():
    return _MockDiscreteIO(vocab_size=100, n_stream=1, modality="text")


def _build_audio_io():
    return _MockDiscreteIO(vocab_size=400, n_stream=4, modality="audio")


def _build_continuous_io():
    return _MockContinuousIO(feat_dim=80)


def _make_vocab_meta(text_io, audio_io):
    """Build a vocab_meta dict matching the new ParallelLLM.from_pretrained API.

    Mirrors the layout SpeechLMJobTemplate._build_vocabulary produces: 256
    special tokens, then text, then discrete_audio. Text covers the original
    pretrained vocab (100), audio becomes the multimodal vocab range.
    """
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

    text_start = 256
    vocab.extend(text_io.get_vocabulary())
    intervals["text"] = [
        (text_start + s, text_start + e) for s, e in text_io.get_stream_interval()
    ]
    text_end = len(vocab)

    mm_start = len(vocab)
    vocab.extend(audio_io.get_vocabulary())
    intervals["discrete_audio"] = [
        (mm_start + s, mm_start + e) for s, e in audio_io.get_stream_interval()
    ]
    mm_end = len(vocab)

    vocab_size = len(vocab)
    return {
        "vocab": vocab,
        "vocab_intervals": intervals,
        "vocab_weight": torch.ones(vocab_size, dtype=torch.float32),
        "vocab_size": vocab_size,
        "mm_start": mm_start,
        "mm_end": mm_end,
        "text_start": text_start,
        "text_end": text_end,
        "num_stream": max(text_io.num_stream(), audio_io.num_stream()),
    }


@pytest.fixture
def model_components():
    """Create model components for parallel LLM tests."""
    text_io = _build_text_io()
    audio_io = _build_audio_io()
    multimodal_io = nn.ModuleDict({"text": text_io, "discrete_audio": audio_io})
    vocab_meta = _make_vocab_meta(text_io, audio_io)
    return multimodal_io, vocab_meta


@pytest.fixture
def parallel_model(model_components):
    """Build a ParallelLLM model using mock architecture."""
    from espnet2.speechlm.model.speechlm.lm.parallel import (
        build_parallel_hf_class,
    )

    multimodal_io, vocab_meta = model_components
    cls = build_parallel_hf_class("mock-model")
    model = cls.from_pretrained(
        "mock-model",
        multimodal_io=multimodal_io,
        vocab_meta=vocab_meta,
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
        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )

        cls = build_parallel_hf_class("mock-model")
        assert issubclass(cls, _MockHFModel)


# ---------------------------------------------------------------------------
# from_pretrained
# ---------------------------------------------------------------------------
class TestFromPretrained:
    def test_rebuilds_embeddings(self, parallel_model, model_components):
        _, vocab_meta = model_components
        expected_vocab_size = vocab_meta["vocab_size"]
        assert parallel_model.model.embed_tokens.weight.shape[0] == expected_vocab_size
        assert parallel_model.lm_head.weight.shape[0] == expected_vocab_size

    def test_stream_emb(self, parallel_model):
        assert hasattr(parallel_model, "stream_emb")
        assert parallel_model.stream_emb.weight.shape[0] == parallel_model.num_stream
        assert (
            parallel_model.stream_emb.weight.shape[1]
            == parallel_model.config.hidden_size
        )

    def test_multimodal_vocab_range(self, parallel_model, model_components):
        _, vocab_meta = model_components
        assert parallel_model.multimodal_vocab_range == (
            vocab_meta["mm_start"],
            vocab_meta["mm_end"],
        )

    def test_vocab_weight_buffer(self, parallel_model, model_components):
        _, vocab_meta = model_components
        assert hasattr(parallel_model, "vocab_weight")
        assert parallel_model.vocab_weight.shape == (vocab_meta["vocab_size"],)

    def test_vocab_and_intervals_exposed(self, parallel_model, model_components):
        _, vocab_meta = model_components
        assert parallel_model.vocab == vocab_meta["vocab"]
        assert parallel_model.vocab_intervals == vocab_meta["vocab_intervals"]

    def test_adaptor_for_continuous(self):
        """Continuous IOs should get linear adaptors."""
        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )

        text_io = _build_text_io()
        audio_io = _build_audio_io()
        cont_io = _build_continuous_io()
        multimodal_io = nn.ModuleDict(
            {
                "text": text_io,
                "discrete_audio": audio_io,
                "continuous_audio": cont_io,
            }
        )
        vocab_meta = _make_vocab_meta(text_io, audio_io)
        cls = build_parallel_hf_class("mock-model")
        model = cls.from_pretrained(
            "mock-model",
            multimodal_io=multimodal_io,
            vocab_meta=vocab_meta,
        )
        assert "continuous_audio" in model.adaptor
        assert isinstance(model.adaptor["continuous_audio"], nn.Linear)
        assert model.adaptor["continuous_audio"].in_features == 80

    def test_tie_word_embeddings(self, model_components):
        """tie_word_embeddings=True shares embed_tokens weight with lm_head."""
        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )

        multimodal_io, vocab_meta = model_components
        cls = build_parallel_hf_class("mock-model")
        model = cls.from_pretrained(
            "mock-model",
            multimodal_io=multimodal_io,
            vocab_meta=vocab_meta,
            tie_word_embeddings=True,
        )
        assert model.lm_head.weight is model.model.embed_tokens.weight

    def test_z_loss_weight_stored(self, model_components):
        """z_loss_weight kwarg is stored on the model for _loss to read."""
        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )

        multimodal_io, vocab_meta = model_components
        cls = build_parallel_hf_class("mock-model")
        model = cls.from_pretrained(
            "mock-model",
            multimodal_io=multimodal_io,
            vocab_meta=vocab_meta,
            z_loss_weight=2.5e-5,
        )
        assert model.z_loss_weight == 2.5e-5

    def test_text_vocab_size_mismatch_raises(self, model_components):
        """text_end - text_start must equal the pretrained vocab_size (100)."""
        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )

        multimodal_io, vocab_meta = model_components
        bad = dict(vocab_meta)
        bad["text_end"] = bad["text_start"] + 50  # mismatches mock's vocab_size=100
        cls = build_parallel_hf_class("mock-model")
        with pytest.raises(ValueError, match="must equal original vocab size"):
            cls.from_pretrained(
                "mock-model",
                multimodal_io=multimodal_io,
                vocab_meta=bad,
            )

    def test_flash_attention_assertion(self, model_components, monkeypatch):
        """Non-flash attn_implementation should fail the assertion."""
        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )

        multimodal_io, vocab_meta = model_components

        class _Cfg(_MockConfig):
            _attn_implementation = "sdpa"

        import transformers

        monkeypatch.setattr(
            transformers.AutoConfig, "from_pretrained", lambda *a, **kw: _Cfg()
        )

        class _BadHFModel(_MockHFModel):
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls(_Cfg())

        monkeypatch.setattr(transformers, "MockModel", _BadHFModel)

        cls = build_parallel_hf_class("mock-model")
        with pytest.raises(AssertionError, match="Flash Attention"):
            cls.from_pretrained(
                "mock-model",
                multimodal_io=multimodal_io,
                vocab_meta=vocab_meta,
            )


# ---------------------------------------------------------------------------
# forward / _loss
# ---------------------------------------------------------------------------
class TestForward:
    def test_returns_scalar_loss(self, parallel_model):
        batch_size, seq_len = 2, 10
        num_stream = parallel_model.num_stream
        seqs = torch.zeros(batch_size, seq_len, num_stream, dtype=torch.long)
        loss_masks = torch.ones(batch_size, seq_len, num_stream)

        parallel_model.train()
        parallel_model.reset_loss_stats()
        out = parallel_model(seqs=seqs, loss_masks=loss_masks)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0

    def test_loss_stats_populated(self, parallel_model):
        batch_size, seq_len = 2, 10
        num_stream = parallel_model.num_stream
        seqs = torch.zeros(batch_size, seq_len, num_stream, dtype=torch.long)
        loss_masks = torch.ones(batch_size, seq_len, num_stream)

        parallel_model.train()
        parallel_model.reset_loss_stats()
        parallel_model(seqs=seqs, loss_masks=loss_masks)
        stats = parallel_model._loss_stats
        assert "ce_loss" in stats
        assert "count" in stats

    def test_reset_loss_stats_clears(self, parallel_model):
        parallel_model.reset_loss_stats()
        assert parallel_model._loss_stats == {}


class TestLoss:
    def test_loss_scalar(self, parallel_model):
        """_loss returns a scalar tensor.

        The stub Liger returns zero so we only check shape/flow, not
        numeric correctness.
        """
        batch_size, seq_len = 1, 6
        num_stream = parallel_model.num_stream
        hidden_dim = parallel_model.config.hidden_size

        last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
        input_ids = torch.zeros(batch_size, seq_len, num_stream, dtype=torch.long)
        loss_mask = torch.ones(batch_size, seq_len, num_stream)

        parallel_model.train()
        parallel_model.reset_loss_stats()
        loss = parallel_model._loss(
            (last_hidden_state, None),
            (input_ids, loss_mask),
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_loss_scale_applied(self, parallel_model):
        """When scale is passed, loss is multiplied by it."""
        batch_size, seq_len = 1, 4
        num_stream = parallel_model.num_stream
        hidden_dim = parallel_model.config.hidden_size

        last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
        input_ids = torch.zeros(batch_size, seq_len, num_stream, dtype=torch.long)
        loss_mask = torch.ones(batch_size, seq_len, num_stream)

        parallel_model.train()
        parallel_model.reset_loss_stats()
        loss = parallel_model._loss(
            (last_hidden_state, None),
            (input_ids, loss_mask),
            scale=2.5,
        )
        # With the zero-stub Liger, loss is 0 * 2.5 = 0, but the tensor exists.
        assert loss.shape == ()

    def test_loss_stats_accumulate_across_calls(self, parallel_model):
        """Repeated _loss calls accumulate into self._loss_stats."""
        batch_size, seq_len = 1, 4
        num_stream = parallel_model.num_stream
        hidden_dim = parallel_model.config.hidden_size

        last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
        input_ids = torch.zeros(batch_size, seq_len, num_stream, dtype=torch.long)
        loss_mask = torch.ones(batch_size, seq_len, num_stream)

        parallel_model.train()
        parallel_model.reset_loss_stats()
        parallel_model._loss((last_hidden_state, None), (input_ids, loss_mask))
        count_after_one = float(parallel_model._loss_stats["count"])
        parallel_model._loss((last_hidden_state, None), (input_ids, loss_mask))
        count_after_two = float(parallel_model._loss_stats["count"])
        assert count_after_two == 2 * count_after_one

    def test_loss_stream_emb_dtensor(self, parallel_model):
        """If stream_emb.weight has .full_tensor(), it's materialized."""
        batch_size, seq_len = 1, 4
        num_stream = parallel_model.num_stream
        hidden_dim = parallel_model.config.hidden_size

        last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
        input_ids = torch.zeros(batch_size, seq_len, num_stream, dtype=torch.long)
        loss_mask = torch.ones(batch_size, seq_len, num_stream)

        # Wrap the stream_emb weight with a .full_tensor() shim.
        real_weight = parallel_model.stream_emb.weight.data

        class _DTensorShim:
            full_tensor_calls = 0

            def __init__(self, tensor):
                self._tensor = tensor
                self.dtype = tensor.dtype
                self.device = tensor.device

            def full_tensor(self):
                _DTensorShim.full_tensor_calls += 1
                return self._tensor

            def to(self, *args, **kwargs):
                return self._tensor.to(*args, **kwargs)

        shim = _DTensorShim(real_weight)
        # monkey-patch the stream_emb.weight attribute with a DTensor shim
        object.__setattr__(parallel_model.stream_emb, "weight", shim)

        parallel_model.train()
        parallel_model.reset_loss_stats()
        parallel_model._loss((last_hidden_state, None), (input_ids, loss_mask))
        assert _DTensorShim.full_tensor_calls >= 1


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
        assert embeds.shape == (
            batch_size,
            seq_len,
            parallel_model.config.hidden_size,
        )

    def test_embed_discrete_features_with_indices(self, parallel_model):
        """Discrete IO with indices/feats/lengths triggers encode_batch path."""
        batch_size, seq_len = 1, 10
        num_stream = parallel_model.num_stream
        input_ids = torch.zeros(batch_size, seq_len, num_stream, dtype=torch.long)
        kwargs = {
            "seqs": input_ids,
            "discrete_audio_indices": torch.tensor([[0, 2, 3]]),
            "discrete_audio_feats": torch.randn(1, 3, 80),
            "discrete_audio_lengths": torch.tensor([3]),
        }
        embeds = parallel_model._embed(input_ids, kwargs)
        assert embeds.shape == (
            batch_size,
            seq_len,
            parallel_model.config.hidden_size,
        )

    def test_embed_continuous_features(self):
        """Continuous features projected through adaptor."""
        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )

        text_io = _build_text_io()
        audio_io = _build_audio_io()
        cont_io = _build_continuous_io()
        multimodal_io = nn.ModuleDict(
            {
                "text": text_io,
                "discrete_audio": audio_io,
                "continuous_audio": cont_io,
            }
        )
        vocab_meta = _make_vocab_meta(text_io, audio_io)
        cls = build_parallel_hf_class("mock-model")
        model = cls.from_pretrained(
            "mock-model",
            multimodal_io=multimodal_io,
            vocab_meta=vocab_meta,
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
        assert embeds.shape == (
            batch_size,
            seq_len,
            model.config.hidden_size,
        )


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
        assert parallel_model.assistant_token.shape == (
            1,
            1,
            parallel_model.num_stream,
        )


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
        assert (tokens >= 0).all()
        assert (tokens < len(parallel_model.vocab)).all()
