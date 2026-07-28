"""Tests for espnet2/speechlm/model/speechlm/lm/parallel_pp.py.

Only pure-Python helpers are covered here: ``_prune_to_stage`` and
``_merge_router_logits``, plus a sanity check on
``build_parallel_pp_hf_class`` returning a subclass of the non-PP
parallel model.

The heavy paths — ``from_pretrained``, ``_empty_init``, ``forward``
(incl. ``_forward_first_stage`` / ``_forward_middle_stage`` /
``_forward_last_stage``), and ``_run_decoder_layers`` — all require
``torch.distributed`` (``dist.broadcast``), real HF decoder layers, or
meta-device materialization. They are intentionally left to integration
testing on a multi-GPU host.

The whole file is skipped (via the ``test/espnet2/speechlm/model/``
``conftest.py``'s ``collect_ignore_glob``) when
``liger_kernel.ops.fused_linear_cross_entropy`` is not importable
(transitive via ``parallel.py`` → ``loss.py``).
"""

import pytest
import torch
import torch.nn as nn


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
        out.get = lambda key, default=None: default
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
        return cls(_MockConfig())


@pytest.fixture(autouse=True)
def _patch_transformers(monkeypatch):
    """Patch transformers for ``build_parallel_pp_hf_class`` via monkeypatch.

    Routes every mutation through pytest's ``monkeypatch`` so teardown
    is guaranteed and global state cannot leak into unrelated tests.
    """
    import transformers

    monkeypatch.setattr(transformers, "MockModel", _MockHFModel, raising=False)
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *a, **kw: _MockConfig()),
    )
    yield


# ---------------------------------------------------------------------------
# _merge_router_logits
# ---------------------------------------------------------------------------
class TestMergeRouterLogits:
    def test_both_none(self):
        from espnet2.speechlm.model.speechlm.lm.parallel_pp import (
            _merge_router_logits,
        )

        assert _merge_router_logits(None, None) is None

    def test_prev_none(self):
        from espnet2.speechlm.model.speechlm.lm.parallel_pp import (
            _merge_router_logits,
        )

        local = torch.zeros(3, 4)
        assert _merge_router_logits(None, local) is local

    def test_local_none(self):
        from espnet2.speechlm.model.speechlm.lm.parallel_pp import (
            _merge_router_logits,
        )

        prev = torch.zeros(3, 4)
        assert _merge_router_logits(prev, None) is prev

    def test_both_tensors_concat_dim0(self):
        from espnet2.speechlm.model.speechlm.lm.parallel_pp import (
            _merge_router_logits,
        )

        prev = torch.ones(2, 4)
        local = torch.zeros(3, 4)
        out = _merge_router_logits(prev, local)
        assert out.shape == (5, 4)
        assert torch.equal(out[:2], prev)
        assert torch.equal(out[2:], local)


# ---------------------------------------------------------------------------
# build_parallel_pp_hf_class
# ---------------------------------------------------------------------------
class TestBuildParallelPPClass:
    def test_subclass_of_parallel_llm(self):
        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )
        from espnet2.speechlm.model.speechlm.lm.parallel_pp import (
            build_parallel_pp_hf_class,
        )

        pp_cls = build_parallel_pp_hf_class("mock-model")
        base_cls = build_parallel_hf_class("mock-model")
        # pp_cls inherits from an independently-built parallel class that is
        # itself a subclass of _MockHFModel. Walk the MRO to confirm the HF
        # base is in both class hierarchies.
        assert issubclass(pp_cls, _MockHFModel)
        assert issubclass(base_cls, _MockHFModel)


# ---------------------------------------------------------------------------
# _prune_to_stage
# ---------------------------------------------------------------------------
def _make_toy_pp_model(num_layers=4):
    """Minimal nn.Module with the attribute shape ``_prune_to_stage`` expects."""

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(num_layers)])
            self.embed_tokens = nn.Embedding(10, 4)
            self.norm = nn.LayerNorm(4)

    class _Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.lm_head = nn.Linear(4, 10, bias=False)
            self.stream_emb = nn.Embedding(2, 4)
            self.multimodal_io_dict = nn.ModuleDict({"text": nn.Linear(1, 1)})
            self.adaptor = nn.ModuleDict({})
            self.register_buffer("vocab_weight", torch.ones(10))

    return _Outer()


def _get_prune_fn():
    from espnet2.speechlm.model.speechlm.lm.parallel_pp import (
        build_parallel_pp_hf_class,
    )

    pp_cls = build_parallel_pp_hf_class("mock-model")
    return pp_cls._prune_to_stage


class TestPruneToStage:
    def test_non_local_layers_become_identity(self):
        prune = _get_prune_fn()
        model = _make_toy_pp_model(num_layers=4)
        originals = list(model.model.layers)
        prune(
            model, layer_start=1, layer_end=3, is_first_stage=False, is_last_stage=False
        )
        assert isinstance(model.model.layers[0], nn.Identity)
        assert isinstance(model.model.layers[3], nn.Identity)
        # Layers 1 and 2 kept their original module identity.
        assert model.model.layers[1] is originals[1]
        assert model.model.layers[2] is originals[2]

    def test_middle_stage_strips_boundary_modules(self):
        prune = _get_prune_fn()
        model = _make_toy_pp_model(num_layers=4)
        prune(model, 1, 3, is_first_stage=False, is_last_stage=False)
        assert model.model.embed_tokens is None
        assert model.multimodal_io_dict is None
        assert model.adaptor is None
        assert model.lm_head is None
        assert model.stream_emb is None
        assert model.model.norm is None
        assert model.vocab_weight is None

    def test_first_stage_keeps_embed(self):
        prune = _get_prune_fn()
        model = _make_toy_pp_model(num_layers=4)
        prune(model, 0, 2, is_first_stage=True, is_last_stage=False)
        assert model.model.embed_tokens is not None
        assert model.multimodal_io_dict is not None
        assert model.adaptor is not None
        # Not last stage: lm_head, stream_emb, norm removed.
        assert model.lm_head is None
        assert model.stream_emb is None
        assert model.model.norm is None

    def test_last_stage_keeps_head(self):
        prune = _get_prune_fn()
        model = _make_toy_pp_model(num_layers=4)
        prune(model, 2, 4, is_first_stage=False, is_last_stage=True)
        assert model.lm_head is not None
        assert model.stream_emb is not None
        assert model.model.norm is not None
        # Not first stage: embed / IO dict / adaptor removed.
        assert model.model.embed_tokens is None
        assert model.multimodal_io_dict is None
        assert model.adaptor is None

    def test_single_stage_keeps_everything(self):
        prune = _get_prune_fn()
        model = _make_toy_pp_model(num_layers=4)
        prune(model, 0, 4, is_first_stage=True, is_last_stage=True)
        # All layers local → no Identity swaps.
        for layer in model.model.layers:
            assert not isinstance(layer, nn.Identity)
        # Both first and last: nothing nulled.
        assert model.model.embed_tokens is not None
        assert model.lm_head is not None
        assert model.stream_emb is not None
        assert model.model.norm is not None


# ---------------------------------------------------------------------------
# forward: pp_degree==1 branch delegates to ParallelLLM.forward
# ---------------------------------------------------------------------------
class TestForwardSingleStage:
    """Forward path when there is only one pipeline stage.

    When pp_degree == 1 there is only one stage, so forward() just
    delegates to the parent ParallelLLM.forward(**kwargs). We reuse
    the test_parallel mock infrastructure by re-classing a ParallelLLM
    instance into ParallelPPLLM and setting the stage metadata by hand
    — this avoids the full from_pretrained path which needs
    torch.distributed.
    """

    def _build_single_stage_model(self, monkeypatch):
        # Fake Liger is already in sys.modules via /tmp launcher when this
        # file is exercised with liger present. Also monkeypatch to avoid
        # the real kernel path inside _loss.
        import torch

        class _FakeLigerFn:
            @staticmethod
            def apply(*args):
                return (
                    torch.zeros((), requires_grad=True),
                    torch.zeros(()),
                    torch.zeros(()),
                )

        monkeypatch.setattr(
            "espnet2.speechlm.model.speechlm.lm.loss."
            "LigerFusedLinearCrossEntropyFunction",
            _FakeLigerFn,
        )

        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )
        from espnet2.speechlm.model.speechlm.lm.parallel_pp import (
            build_parallel_pp_hf_class,
        )

        # Build a regular ParallelLLM first (no distributed).
        text_io = _make_text_io()
        audio_io = _make_audio_io()
        multimodal_io = nn.ModuleDict({"text": text_io, "discrete_audio": audio_io})
        vocab_meta = _make_vocab_meta(text_io, audio_io)

        parallel_cls = build_parallel_hf_class("mock-model")
        model = parallel_cls.from_pretrained(
            "mock-model",
            multimodal_io=multimodal_io,
            vocab_meta=vocab_meta,
        )

        # Re-class to the PP subclass and stamp the minimal PP metadata
        # the forward() branch check reads.
        pp_cls = build_parallel_pp_hf_class("mock-model")
        model.__class__ = pp_cls
        model.pp_rank = 0
        model.pp_degree = 1
        model.is_first_stage = True
        model.is_last_stage = True
        model.stage_idx = 0
        model.num_virtual_stages = 1
        model.z_loss_weight = 0.0
        return model

    def test_forward_delegates_to_super(self, monkeypatch):
        import torch

        model = self._build_single_stage_model(monkeypatch)
        batch_size, seq_len = 1, 5
        num_stream = model.num_stream
        seqs = torch.zeros(batch_size, seq_len, num_stream, dtype=torch.long)
        loss_masks = torch.ones(batch_size, seq_len, num_stream)
        model.train()
        model.reset_loss_stats()
        out = model(seqs=seqs, loss_masks=loss_masks)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0


# ---------------------------------------------------------------------------
# Mock IOs + vocab_meta helper (used by TestForwardSingleStage)
# ---------------------------------------------------------------------------
from espnet2.speechlm.model.speechlm.multimodal_io.abs_io import AbsIO  # noqa


class _MockDiscreteIO(AbsIO):
    """Minimal discrete IO mirroring the one in test_parallel.py."""

    def __init__(self, vocab_size=100, n_stream=1, modality="text"):
        super().__init__(modality=modality, is_discrete=True)
        self._vocab_size = vocab_size
        self._n_stream = n_stream

    def num_stream(self):
        return self._n_stream

    def get_vocabulary(self):
        return [f"{self.modality}_tok_{i}" for i in range(self._vocab_size)]

    def get_stream_interval(self):
        per = self._vocab_size // self._n_stream
        return [(i * per, (i + 1) * per) for i in range(self._n_stream)]

    def get_stream_weight(self):
        return [1.0] * self._n_stream

    def encode_batch(self, feats, lengths):
        return torch.zeros(feats.shape[0], 3, self._n_stream, dtype=torch.long)

    def copy_for_worker(self):
        return self

    def feature_dim(self):
        return None

    def dummy_forward(self, ref_tensor=None):
        return torch.zeros(1, requires_grad=True)


def _make_text_io():
    return _MockDiscreteIO(vocab_size=100, n_stream=1, modality="text")


def _make_audio_io():
    return _MockDiscreteIO(vocab_size=400, n_stream=4, modality="audio")


def _make_vocab_meta(text_io, audio_io):
    vocab = ["<|pad|>"] + [f"<|sp_{i}|>" for i in range(1, 256)]
    text_start = 256
    vocab.extend(text_io.get_vocabulary())
    text_end = len(vocab)
    mm_start = len(vocab)
    vocab.extend(audio_io.get_vocabulary())
    mm_end = len(vocab)
    intervals = {
        "special_token": [(0, 256)],
        "text": [
            (text_start + s, text_start + e) for s, e in text_io.get_stream_interval()
        ],
        "discrete_audio": [
            (mm_start + s, mm_start + e) for s, e in audio_io.get_stream_interval()
        ],
    }
    return {
        "vocab": vocab,
        "vocab_intervals": intervals,
        "vocab_weight": torch.ones(len(vocab), dtype=torch.float32),
        "vocab_size": len(vocab),
        "mm_start": mm_start,
        "mm_end": mm_end,
        "text_start": text_start,
        "text_end": text_end,
        "num_stream": max(text_io.num_stream(), audio_io.num_stream()),
    }
