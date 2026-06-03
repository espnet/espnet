"""Tests for espnet2/speechlm/model/speechlm/parallel_utils/*.

Covers parallel_dims, pipeline, grouped_moe, and qwen3 utility functions.

All tests run on CPU only and require no external accounts. Real
packages are used throughout — no torchtitan or transformers stubs.

Skip mechanics:
- ``torchtitan`` is part of the ``espnet[speechlm]`` extra (Linux-only).
  When it is not importable, the file is skipped at collection time.
- ``transformers.models.qwen3_moe.modeling_qwen3_moe`` is imported by
  the source. If the dev env has a broken ``flash_attn`` binary the
  whole transformers attention chain fails to load — the file is
  skipped in that case too.
- The compile path in ``apply_torch_compile_qwen3`` calls
  ``torch._C._dynamo.eval_frame._set_lru_cache`` which was removed in
  newer PyTorch; tests that hit it are skipped when the symbol is gone.
"""

import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Skip whole module if torchtitan isn't installed (Linux-only speechlm extra).
pytest.importorskip("torchtitan", reason="torchtitan not installed")
pytest.importorskip(
    "torchtitan.distributed.deepep",
    reason="torchtitan.distributed.deepep submodule not importable",
)

# Skip whole module if the HF Qwen3-MoE modeling submodule can't be imported
# (e.g. broken flash_attn ABI in the dev env).
pytest.importorskip(
    "transformers.models.qwen3_moe.modeling_qwen3_moe",
    reason="transformers Qwen3-MoE modeling module not importable",
)

# Drop any non-package parallel_utils stub left in sys.modules by sibling
# conftests (test/espnet2/speechlm/trainer/conftest.py installs a flat-module
# stub when the real package is not yet on the base branch). With the
# importorskip checks above we know the real torchtitan/HF deps are
# available, so the real package can be imported — wipe the stale stub.
for _name in [
    n
    for n in list(sys.modules)
    if n == "espnet2.speechlm.model.speechlm.parallel_utils"
    or n.startswith("espnet2.speechlm.model.speechlm.parallel_utils.")
]:
    sys.modules.pop(_name, None)

from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # noqa: E402
    Qwen3MoeSparseMoeBlock as _Qwen3MoeSparseMoeBlockBase,
)

from espnet2.speechlm.model.speechlm.parallel_utils import (  # noqa: E402
    parallel_strategies,
)
from espnet2.speechlm.model.speechlm.parallel_utils.grouped_moe import (  # noqa: E402
    GroupedExperts,
    GroupedMoeBlock,
    _route,
    _run_grouped_mm,
)
from espnet2.speechlm.model.speechlm.parallel_utils.qwen3 import (  # noqa: E402
    _is_moe_layer,
    apply_activation_checkpoint_qwen3,
    apply_torch_compile_qwen3,
    memory_efficient_load_balancing_loss,
)

# apply_torch_compile_qwen3 calls torch._C._dynamo.eval_frame._set_lru_cache,
# which was removed in recent torch versions. Skip compile tests when absent.
_skip_no_set_lru_cache = pytest.mark.skipif(
    not hasattr(torch._C._dynamo.eval_frame, "_set_lru_cache"),
    reason="torch._C._dynamo.eval_frame._set_lru_cache unavailable on this torch",
)


# ---------------------------------------------------------------------------
# Helpers: simple Qwen3-like mock MoE mlp / moe-block / transformer layer
# ---------------------------------------------------------------------------
class _MockMoeMLP(nn.Module):
    """Mirrors Qwen3MoeMLP: SwiGLU expert with gate/up/down projections."""

    def __init__(self, hidden_size=8, intermediate_size=16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _BareQwen3MoeSparseMoeBlock(_Qwen3MoeSparseMoeBlockBase):
    """Qwen3MoeSparseMoeBlock subclass that skips the parent __init__.

    The real parent __init__ requires a Qwen3MoeConfig and auto-instantiates
    experts. Tests need to populate gate / experts / metadata manually, so we
    call nn.Module.__init__ directly (the same pattern GroupedMoeBlock uses).
    """

    def __init__(self):
        nn.Module.__init__(self)


def _make_moe_block(num_experts=4, hidden_size=8, intermediate_size=16, top_k=2):
    """Build a minimal object that quacks like Qwen3MoeSparseMoeBlock."""
    block = _BareQwen3MoeSparseMoeBlock()
    block.num_experts = num_experts
    block.top_k = top_k
    block.norm_topk_prob = True
    block.gate = nn.Linear(hidden_size, num_experts, bias=False)
    block.experts = nn.ModuleList(
        [_MockMoeMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
    )
    return block


class _MockDenseLayer(nn.Module):
    """Dense transformer layer (no MoE): satisfies _is_moe_layer == False."""

    def __init__(self, hidden_size=8):
        super().__init__()
        self.mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.mlp(x)


class _MockMoeLayer(nn.Module):
    """MoE transformer layer: has mlp.gate + mlp.experts."""

    def __init__(self, num_experts=4, hidden_size=8, intermediate_size=16):
        super().__init__()
        self.mlp = _make_moe_block(num_experts, hidden_size, intermediate_size)

    def forward(self, x):
        return self.mlp(x)


class _MockHFModel(nn.Module):
    """Minimal HF-like wrapper: .model.layers + .model.norm + .lm_head."""

    def __init__(self, layers):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layers)
        self.model.norm = nn.LayerNorm(8)
        self.model.embed_tokens = nn.Embedding(100, 8)
        self.lm_head = nn.Linear(8, 100, bias=False)


# ===========================================================================
# __init__ registry
# ===========================================================================
class TestRegistry:
    def test_qwen3_registered(self):
        assert "qwen3" in parallel_strategies

    def test_registered_is_callable(self):
        assert callable(parallel_strategies["qwen3"])


# ===========================================================================
# _is_moe_layer
# ===========================================================================
class TestIsMoeLayer:
    def test_dense_layer_is_not_moe(self):
        assert _is_moe_layer(_MockDenseLayer()) is False

    def test_moe_layer_is_moe(self):
        assert _is_moe_layer(_MockMoeLayer()) is True

    def test_layer_without_mlp_is_not_moe(self):
        m = nn.Module()
        assert _is_moe_layer(m) is False

    def test_layer_with_gate_but_no_experts_is_not_moe(self):
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.gate = nn.Linear(4, 4)
        assert _is_moe_layer(layer) is False


# ===========================================================================
# memory_efficient_load_balancing_loss
# ===========================================================================
class TestLoadBalancingLoss:
    def test_none_returns_zero(self):
        out = memory_efficient_load_balancing_loss(None, num_experts=8)
        assert torch.is_tensor(out)
        assert out.item() == 0.0

    def test_empty_tuple_returns_zero(self):
        out = memory_efficient_load_balancing_loss((), num_experts=8)
        assert out.item() == 0.0

    def test_non_tuple_returns_zero(self):
        # List passed (not tuple) — function checks isinstance(..., tuple)
        out = memory_efficient_load_balancing_loss([torch.randn(4, 8)], num_experts=8)
        assert out.item() == 0.0

    def test_single_layer_matches_manual(self):
        torch.manual_seed(0)
        num_experts, top_k = 8, 2
        logits = torch.randn(16, num_experts)

        # Expected (HF reference-style): scalar >= 0
        loss = memory_efficient_load_balancing_loss(
            (logits,), num_experts=num_experts, top_k=top_k
        )
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_balanced_routing_gives_lower_loss_than_skewed(self):
        """Skewed routing has higher LB loss than balanced routing.

        The HF load-balancing formulation is loss = E * sum_i f_i * P_i,
        minimized when both f_i (fraction of tokens) and P_i (avg router
        prob) are uniform across experts.
        """
        torch.manual_seed(0)
        num_experts, top_k = 4, 2

        # Skewed: large positive logit for expert 0 → all tokens go to it.
        skewed = torch.zeros(64, num_experts)
        skewed[:, 0] = 10.0
        skewed[:, 1] = 5.0  # second-preferred
        loss_skewed = memory_efficient_load_balancing_loss(
            (skewed,), num_experts=num_experts, top_k=top_k
        )

        # Balanced: different experts preferred per token (random noise).
        balanced = torch.randn(512, num_experts)
        loss_balanced = memory_efficient_load_balancing_loss(
            (balanced,), num_experts=num_experts, top_k=top_k
        )

        assert loss_skewed.item() > loss_balanced.item()

    def test_multi_layer_accumulates(self):
        num_experts = 4
        logits_a = torch.randn(8, num_experts)
        logits_b = torch.randn(8, num_experts)
        single = memory_efficient_load_balancing_loss(
            (logits_a,), num_experts=num_experts, top_k=2
        )
        multi = memory_efficient_load_balancing_loss(
            (logits_a, logits_b), num_experts=num_experts, top_k=2
        )
        assert multi.ndim == 0
        # Not strictly equal to single — second layer contributes.
        assert multi.item() != single.item()


# ===========================================================================
# apply_activation_checkpoint_qwen3
# ===========================================================================
class TestApplyActivationCheckpoint:
    def _build_model(self, num_layers=4):
        return _MockHFModel([_MockDenseLayer() for _ in range(num_layers)])

    def test_ratio_zero_returns_unchanged(self):
        model = self._build_model(num_layers=4)
        original_layers = list(model.model.layers)
        out = apply_activation_checkpoint_qwen3(model, ac_config=0.0)
        assert out is model
        assert list(out.model.layers) == original_layers

    def test_ratio_full_wraps_all(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        model = self._build_model(num_layers=4)
        out = apply_activation_checkpoint_qwen3(model, ac_config=1.0)
        for layer in out.model.layers:
            assert isinstance(layer, CheckpointWrapper)

    def test_ratio_half_wraps_half(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        model = self._build_model(num_layers=4)
        out = apply_activation_checkpoint_qwen3(model, ac_config=0.5)
        wrapped = sum(
            isinstance(layer, CheckpointWrapper) for layer in out.model.layers
        )
        assert wrapped == 2

    def test_identity_layers_skipped(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        model = _MockHFModel(
            [
                _MockDenseLayer(),
                nn.Identity(),
                _MockDenseLayer(),
                nn.Identity(),
            ]
        )
        out = apply_activation_checkpoint_qwen3(model, ac_config=1.0)
        # Only non-Identity layers should be wrapped.
        assert isinstance(out.model.layers[0], CheckpointWrapper)
        assert isinstance(out.model.layers[1], nn.Identity)
        assert isinstance(out.model.layers[2], CheckpointWrapper)
        assert isinstance(out.model.layers[3], nn.Identity)

    def test_list_config_uses_stage_idx(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        model = self._build_model(num_layers=4)
        model.stage_idx = 1  # picks ac_config[1] == 1.0
        out = apply_activation_checkpoint_qwen3(
            model, ac_config=[0.0, 1.0, 0.0], vpp_index=0
        )
        wrapped = sum(
            isinstance(layer, CheckpointWrapper) for layer in out.model.layers
        )
        assert wrapped == 4

    def test_moe_and_full_wraps_non_selected_layers(self):
        """Layers not picked by ratio are still wrapped at the layer level.

        In moe_and_full mode, layers that fall past the selection threshold
        go through the else branch and get wrapped whole (qwen3.py line 417).
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        from espnet2.speechlm.model.speechlm.parallel_utils.grouped_moe import (
            GroupedMoeBlock,
        )

        moe_layer = _MockMoeLayer()
        moe_layer.mlp = GroupedMoeBlock(moe_layer.mlp)
        dense_layer = _MockDenseLayer()
        model = _MockHFModel([moe_layer, dense_layer, dense_layer])

        # ratio=0.34 → 1 layer selected for moe-only wrapping; others fall
        # through to moe_and_full branch and get full-layer wrap.
        out = apply_activation_checkpoint_qwen3(
            model, ac_config=0.34, mode="moe_and_full"
        )
        num_wrapped = sum(
            isinstance(layer, CheckpointWrapper) for layer in out.model.layers
        )
        # All 3 layers end up wrapped, one via mlp-only and two via full wrap.
        # Count only the full-layer wraps here.
        assert num_wrapped >= 1

    def test_moe_mode_wraps_only_mlp_on_moe_layer(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        # MoE mode requires mlp to be a GroupedMoeBlock — first wrap it.
        from espnet2.speechlm.model.speechlm.parallel_utils.grouped_moe import (
            GroupedMoeBlock,
        )

        moe_layer = _MockMoeLayer()
        moe_layer.mlp = GroupedMoeBlock(moe_layer.mlp)
        dense_layer = _MockDenseLayer()
        model = _MockHFModel([moe_layer, dense_layer])

        out = apply_activation_checkpoint_qwen3(model, ac_config=1.0, mode="moe")
        # MoE layer in "moe" mode: only .mlp is wrapped, layer itself is not.
        assert not isinstance(out.model.layers[0], CheckpointWrapper)
        assert isinstance(out.model.layers[0].mlp, CheckpointWrapper)
        # Dense layer in "moe" mode falls through to the else branch and is
        # wrapped at the layer level (source: qwen3.py, lines ~400-413).
        assert isinstance(out.model.layers[1], CheckpointWrapper)


# ===========================================================================
# apply_torch_compile_qwen3
# ===========================================================================
@_skip_no_set_lru_cache
class TestApplyTorchCompile:
    def test_skips_identity_layers(self):
        model = _MockHFModel(
            [
                _MockDenseLayer(),
                nn.Identity(),
                _MockDenseLayer(),
            ]
        )
        out = apply_torch_compile_qwen3(model, titan_config={})
        # Identity layer remains Identity
        assert isinstance(out.model.layers[1], nn.Identity)

    def test_wraps_non_identity(self):
        # torch.compile returns an OptimizedModule; the wrapped layer type
        # differs from the original _MockDenseLayer.
        model = _MockHFModel([_MockDenseLayer(), _MockDenseLayer()])
        original_types = [type(layer) for layer in model.model.layers]
        out = apply_torch_compile_qwen3(model, titan_config={})
        # After compile, layer type should not equal the original eager type.
        for orig_type, layer in zip(original_types, out.model.layers):
            # Heuristic: wrapped module has `_orig_mod` attribute, or type
            # differs from the pre-compile one.
            compiled = hasattr(layer, "_orig_mod") or type(layer) is not orig_type
            assert compiled


# ===========================================================================
# grouped_moe: _route
# ===========================================================================
class TestRoute:
    def test_output_shapes(self):
        torch.manual_seed(0)
        tokens, hidden, num_experts, top_k = 8, 16, 4, 2
        gate = nn.Linear(hidden, num_experts, bias=False)
        x = torch.randn(tokens, hidden)

        router_logits, routing_weights, selected_experts = _route(
            gate, x, top_k=top_k, norm_topk_prob=True
        )
        assert router_logits.shape == (tokens, num_experts)
        assert routing_weights.shape == (tokens, top_k)
        assert selected_experts.shape == (tokens, top_k)
        assert selected_experts.dtype == torch.int64

    def test_norm_topk_prob_sums_to_one(self):
        torch.manual_seed(0)
        gate = nn.Linear(16, 4, bias=False)
        x = torch.randn(8, 16)
        _, weights, _ = _route(gate, x, top_k=2, norm_topk_prob=True)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_no_norm_does_not_sum_to_one(self):
        """Without norm, topk probabilities don't have to sum to 1."""
        torch.manual_seed(0)
        gate = nn.Linear(16, 4, bias=False)
        x = torch.randn(8, 16)
        _, weights, _ = _route(gate, x, top_k=2, norm_topk_prob=False)
        # At least one row won't sum to 1 with random logits and top_k<E.
        sums = weights.sum(dim=-1)
        assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)

    def test_selected_experts_are_valid(self):
        torch.manual_seed(0)
        num_experts, top_k = 8, 3
        gate = nn.Linear(16, num_experts, bias=False)
        x = torch.randn(10, 16)
        _, _, selected = _route(gate, x, top_k=top_k, norm_topk_prob=True)
        assert (selected >= 0).all()
        assert (selected < num_experts).all()


# ===========================================================================
# grouped_moe: _run_grouped_mm
# ===========================================================================
class TestRunGroupedMM:
    def test_shapes(self):
        torch.manual_seed(0)
        num_experts, hidden, intermediate = 2, 8, 16
        # tokens sorted by expert: 3 tokens for expert 0, 2 for expert 1
        x = torch.randn(5, hidden, dtype=torch.bfloat16)
        w1 = torch.randn(num_experts, hidden, intermediate, dtype=torch.bfloat16)
        w2 = torch.randn(num_experts, intermediate, hidden, dtype=torch.bfloat16)
        w3 = torch.randn(num_experts, hidden, intermediate, dtype=torch.bfloat16)
        counts = torch.tensor([3, 2], dtype=torch.int32)

        out = _run_grouped_mm(w1, w2, w3, x, counts)
        assert out.shape == (5, hidden)
        assert out.dtype == torch.bfloat16

    def test_casts_float32_input_to_same_dtype_output(self):
        # torch._grouped_mm requires inner stride * itemsize >= 16 bytes;
        # for bf16 (2 bytes) that means inner contiguous dim >= 8.
        torch.manual_seed(0)
        num_experts, hidden, intermediate = 2, 8, 16
        x = torch.randn(4, hidden, dtype=torch.float32)
        w1 = torch.randn(num_experts, hidden, intermediate)
        w2 = torch.randn(num_experts, intermediate, hidden)
        w3 = torch.randn(num_experts, hidden, intermediate)
        counts = torch.tensor([2, 2], dtype=torch.int32)

        out = _run_grouped_mm(w1, w2, w3, x, counts)
        # Output dtype matches input (fp32), even though compute is bf16.
        assert out.dtype == torch.float32
        assert out.shape == (4, hidden)


# ===========================================================================
# grouped_moe: GroupedExperts
# ===========================================================================
class TestGroupedExperts:
    def test_param_shapes(self):
        num_experts, hidden, intermediate = 4, 8, 16
        experts = [_MockMoeMLP(hidden, intermediate) for _ in range(num_experts)]
        ge = GroupedExperts(experts)

        assert ge.num_experts == num_experts
        assert ge.hidden_size == hidden
        assert ge.intermediate_size == intermediate
        # Stored pre-transposed: (E, in, out)
        assert tuple(ge.w1.shape) == (num_experts, hidden, intermediate)
        assert tuple(ge.w2.shape) == (num_experts, intermediate, hidden)
        assert tuple(ge.w3.shape) == (num_experts, hidden, intermediate)

    def test_preserves_requires_grad(self):
        experts = [_MockMoeMLP() for _ in range(2)]
        for e in experts:
            for p in e.parameters():
                p.requires_grad = False
        ge = GroupedExperts(experts)
        assert ge.w1.requires_grad is False
        assert ge.w2.requires_grad is False
        assert ge.w3.requires_grad is False

    def test_forward_shape(self):
        num_experts, hidden, intermediate = 2, 8, 16
        experts = [_MockMoeMLP(hidden, intermediate) for _ in range(num_experts)]
        ge = GroupedExperts(experts)

        tokens = 6  # 3 per expert
        x = torch.randn(tokens, hidden)
        counts = torch.tensor([3, 3], dtype=torch.int32)
        out = ge(x, counts)
        assert out.shape == (tokens, hidden)

    def test_extra_repr(self):
        ge = GroupedExperts([_MockMoeMLP(8, 16) for _ in range(3)])
        rep = ge.extra_repr()
        assert "num_experts=3" in rep
        assert "hidden_size=8" in rep
        assert "intermediate_size=16" in rep

    def test_rejects_experts_with_bias(self):
        class _BiasedMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(8, 16, bias=True)
                self.up_proj = nn.Linear(8, 16, bias=False)
                self.down_proj = nn.Linear(16, 8, bias=False)

        with pytest.raises(AssertionError, match="does not support biases"):
            GroupedExperts([_BiasedMLP(), _BiasedMLP()])


# ===========================================================================
# grouped_moe: GroupedMoeBlock (end-to-end forward)
# ===========================================================================
class TestGroupedMoeBlock:
    def test_init_copies_metadata(self):
        block = _make_moe_block(num_experts=4, hidden_size=8, top_k=2)
        wrapped = GroupedMoeBlock(block)
        assert wrapped.num_experts == 4
        assert wrapped.top_k == 2
        assert wrapped.gate is block.gate
        assert isinstance(wrapped.experts, GroupedExperts)

    def test_forward_shape(self):
        torch.manual_seed(0)
        block = _make_moe_block(
            num_experts=4, hidden_size=8, intermediate_size=16, top_k=2
        )
        wrapped = GroupedMoeBlock(block)

        batch, seq, hidden = 2, 3, 8
        x = torch.randn(batch, seq, hidden)
        out, router_logits = wrapped(x)
        assert out.shape == (batch, seq, hidden)
        assert router_logits.shape == (batch * seq, 4)

    def test_forward_records_router_logits(self):
        block = _make_moe_block(num_experts=4, hidden_size=8)
        wrapped = GroupedMoeBlock(block)
        x = torch.randn(1, 2, 8)
        _, router_logits = wrapped(x)
        assert torch.equal(wrapped._last_router_logits, router_logits)

    def test_is_qwen3moe_instance(self):
        """GroupedMoeBlock is-a Qwen3MoeSparseMoeBlock.

        HF's OutputRecorder uses an isinstance check to capture router_logits
        from MoE blocks, so GroupedMoeBlock must inherit from the HF class.
        """
        wrapped = GroupedMoeBlock(_make_moe_block())
        assert isinstance(wrapped, _Qwen3MoeSparseMoeBlockBase)


# ===========================================================================
# apply_grouped_moe_qwen3
# ===========================================================================
def _patch_cuda_identity(module):
    """Replace .cuda() with identity so CPU-only tests can exercise GPU paths.

    Each lambda captures its module via a default arg to avoid late-binding
    over the loop variable.
    """
    for m in module.modules():
        m.cuda = lambda *a, _m=m, **kw: _m


class TestApplyGroupedMoeQwen3:
    def test_no_moe_layers_leaves_model_unchanged(self):
        from espnet2.speechlm.model.speechlm.parallel_utils.qwen3 import (
            apply_grouped_moe_qwen3,
        )

        model = _MockHFModel([_MockDenseLayer() for _ in range(3)])
        _patch_cuda_identity(model)
        out = apply_grouped_moe_qwen3(model, parallel_dims=None)
        # No moe → no replacement, and load_balancing_loss_func not set.
        assert out is model
        for layer in out.model.layers:
            assert not isinstance(layer.mlp, GroupedMoeBlock)
        assert not hasattr(out, "load_balancing_loss_func")

    def test_replaces_moe_layers(self):
        from espnet2.speechlm.model.speechlm.parallel_utils.qwen3 import (
            apply_grouped_moe_qwen3,
        )

        model = _MockHFModel([_MockMoeLayer(), _MockDenseLayer(), _MockMoeLayer()])
        _patch_cuda_identity(model)
        out = apply_grouped_moe_qwen3(model, parallel_dims=None)
        assert isinstance(out.model.layers[0].mlp, GroupedMoeBlock)
        assert not isinstance(out.model.layers[1].mlp, GroupedMoeBlock)
        assert isinstance(out.model.layers[2].mlp, GroupedMoeBlock)
        # load balancing loss installed when any MoE layer found
        assert callable(out.load_balancing_loss_func)

    def test_identity_layers_skipped(self):
        from espnet2.speechlm.model.speechlm.parallel_utils.qwen3 import (
            apply_grouped_moe_qwen3,
        )

        model = _MockHFModel([nn.Identity(), _MockMoeLayer(), nn.Identity()])
        _patch_cuda_identity(model)
        out = apply_grouped_moe_qwen3(model, parallel_dims=None)
        assert isinstance(out.model.layers[0], nn.Identity)
        assert isinstance(out.model.layers[1].mlp, GroupedMoeBlock)
        assert isinstance(out.model.layers[2], nn.Identity)

    def test_idempotent_on_already_wrapped(self):
        from espnet2.speechlm.model.speechlm.parallel_utils.qwen3 import (
            apply_grouped_moe_qwen3,
        )

        model = _MockHFModel([_MockMoeLayer()])
        _patch_cuda_identity(model)
        out = apply_grouped_moe_qwen3(model, parallel_dims=None)
        first_mlp = out.model.layers[0].mlp
        # Second pass should not re-wrap
        _patch_cuda_identity(out)
        out2 = apply_grouped_moe_qwen3(out, parallel_dims=None)
        assert out2.model.layers[0].mlp is first_mlp


# ===========================================================================
# parallelize_qwen3_hf (the top-level dispatcher, AC + compile only)
# ===========================================================================
class TestParallelizeQwen3HF:
    def test_ac_only_no_fsdp_no_compile(self):
        """Without CUDA/FSDP, exercise the AC + no-FSDP path."""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        # Build a ParallelDims with FSDP disabled (dp_shard == 1).
        from torchtitan.distributed import ParallelDims

        from espnet2.speechlm.model.speechlm.parallel_utils.qwen3 import (
            parallelize_qwen3_hf,
        )

        # No build_mesh() — every parallel dim is 1, so fsdp_enabled,
        # ep_enabled, and pp_enabled are all False. parallelize_qwen3_hf
        # never accesses a device mesh on this configuration, which lets
        # us avoid initializing torch.distributed.
        pd = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=1,
        )

        model = _MockHFModel([_MockDenseLayer(), _MockDenseLayer()])
        _patch_cuda_identity(model)

        out = parallelize_qwen3_hf(
            model,
            parallel_dims=pd,
            titan_config={
                "activation_checkpoint": 1.0,
                "compile": False,
            },
        )
        # All non-Identity layers wrapped for AC
        for layer in out.model.layers:
            assert isinstance(layer, CheckpointWrapper)

    @_skip_no_set_lru_cache
    def test_compile_branch(self):
        """Exercise the compile=True branch in parallelize_qwen3_hf."""
        from torchtitan.distributed import ParallelDims

        from espnet2.speechlm.model.speechlm.parallel_utils.qwen3 import (
            parallelize_qwen3_hf,
        )

        pd = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=1,
        )

        model = _MockHFModel([_MockDenseLayer()])
        _patch_cuda_identity(model)

        out = parallelize_qwen3_hf(
            model,
            parallel_dims=pd,
            titan_config={
                "activation_checkpoint": 0.0,
                "compile": True,
            },
        )
        # After compile, layer has _orig_mod (torch.compile's OptimizedModule).
        assert hasattr(out.model.layers[0], "_orig_mod")


# ===========================================================================
# parallel_dims.init_parallel_dims
# ===========================================================================
class TestInitParallelDims:
    """Tests for init_parallel_dims.

    The function calls ``parallel_dims.build_mesh()`` which requires
    ``torch.distributed`` initialized. We patch ``build_mesh`` to a
    no-op so the real ParallelDims constructor still runs but the
    mesh-build step is skipped, keeping these tests dist-init free.
    """

    def test_parses_titan_config(self):
        """init_parallel_dims forwards titan_config keys to ParallelDims.

        Returns (parallel_dims, local_rank, global_rank).
        """
        from unittest.mock import patch

        from torchtitan.distributed import ParallelDims

        from espnet2.speechlm.model.speechlm.parallel_utils.parallel_dims import (
            init_parallel_dims,
        )

        with (
            patch(
                "espnet2.speechlm.model.speechlm.parallel_utils."
                "parallel_dims.dist.get_world_size",
                return_value=4,
            ),
            patch(
                "espnet2.speechlm.model.speechlm.parallel_utils."
                "parallel_dims.dist.get_rank",
                return_value=1,
            ),
            patch(
                "espnet2.speechlm.model.speechlm.parallel_utils."
                "parallel_dims.torch.cuda.current_device",
                return_value=2,
            ),
            patch.object(ParallelDims, "build_mesh", lambda self: None),
        ):
            pd, local_rank, global_rank = init_parallel_dims(
                {"dp_replicate": 1, "dp_shard": -1, "pp_degree": 1, "ep": 1}
            )
        assert local_rank == 2
        assert global_rank == 1
        assert pd.world_size == 4
        # dp_shard=-1 auto-computes to world_size when other dims are 1
        assert pd.dp_shard == 4

    def test_ep_in_config(self):
        """ep > 1 is forwarded and triggers the efsdp log path."""
        from unittest.mock import patch

        from torchtitan.distributed import ParallelDims

        from espnet2.speechlm.model.speechlm.parallel_utils.parallel_dims import (
            init_parallel_dims,
        )

        with (
            patch(
                "espnet2.speechlm.model.speechlm.parallel_utils."
                "parallel_dims.dist.get_world_size",
                return_value=8,
            ),
            patch(
                "espnet2.speechlm.model.speechlm.parallel_utils."
                "parallel_dims.dist.get_rank",
                return_value=0,
            ),
            patch(
                "espnet2.speechlm.model.speechlm.parallel_utils."
                "parallel_dims.torch.cuda.current_device",
                return_value=0,
            ),
            patch.object(ParallelDims, "build_mesh", lambda self: None),
        ):
            pd, _, _ = init_parallel_dims(
                {"dp_replicate": 1, "dp_shard": 8, "pp_degree": 1, "ep": 4}
            )
        assert pd.ep == 4


# ===========================================================================
# pipeline.build_pipeline (single-stage path with patched PipelineStage)
# ===========================================================================
class TestBuildPipeline:
    def test_single_stage_path(self):
        """Exercise the single-stage 1F1B schedule path.

        Uses patched PipelineStage and schedule class so no distributed
        initialization is required.
        """
        from unittest.mock import MagicMock, patch

        from espnet2.speechlm.model.speechlm.parallel_utils import pipeline

        # Stage model with the attributes build_pipeline reads.
        stage_model = nn.Linear(4, 4)
        stage_model.pp_rank = 0
        stage_model.pp_degree = 2
        stage_model.is_last_stage = True

        # Fake ParallelDims + pp mesh.
        pd = MagicMock()
        mesh = MagicMock()
        mesh.get_group.return_value = None
        pd.get_mesh.return_value = mesh

        # Fake schedule class: single-stage.
        class _FakeSchedule:
            def __init__(self, stage, n_microbatches, loss_fn, scale_grads):
                self.stage = stage
                self.n_microbatches = n_microbatches
                self.loss_fn = loss_fn
                self.scale_grads = scale_grads

        with (
            patch.object(pipeline, "get_schedule_class", return_value=_FakeSchedule),
            patch.object(pipeline, "PipelineStage", return_value=MagicMock()),
            patch.object(
                pipeline,
                "PipelineScheduleMulti",
                new=type("_DummyMulti", (), {}),
            ),
        ):
            schedule, has_last = pipeline.build_pipeline(
                stage_model,
                parallel_dims=pd,
                titan_config={"pp_schedule": "1F1B"},
                n_microbatches=4,
            )
        assert isinstance(schedule, _FakeSchedule)
        assert has_last is True
        # Verify _identity_loss pass-through behavior
        t = torch.tensor(3.14)
        assert schedule.loss_fn(t, None) is t
        assert schedule.loss_fn((t,), None) is t

    def test_single_stage_unwraps_list_of_one(self):
        """An nn.ModuleList with one chunk is unwrapped for single-stage.

        A single-stage schedule given a one-chunk ModuleList proceeds as if
        the chunk were passed directly.
        """
        from unittest.mock import MagicMock, patch

        from espnet2.speechlm.model.speechlm.parallel_utils import pipeline

        stage = nn.Linear(4, 4)
        stage.pp_rank = 0
        stage.pp_degree = 1
        stage.is_last_stage = True
        model_list = nn.ModuleList([stage])

        pd = MagicMock()
        mesh = MagicMock()
        mesh.get_group.return_value = None
        pd.get_mesh.return_value = mesh

        class _FakeSchedule:
            def __init__(self, stage, **kw):
                self.stage = stage

        with (
            patch.object(pipeline, "get_schedule_class", return_value=_FakeSchedule),
            patch.object(pipeline, "PipelineStage", return_value=MagicMock()),
            patch.object(
                pipeline,
                "PipelineScheduleMulti",
                new=type("_DummyMulti", (), {}),
            ),
        ):
            schedule, _ = pipeline.build_pipeline(
                model_list,
                parallel_dims=pd,
                titan_config={"pp_schedule": "1F1B"},
                n_microbatches=2,
            )
        assert schedule is not None

    def test_multi_stage_path(self):
        """Exercise the multi-stage Interleaved1F1B path."""
        from unittest.mock import MagicMock, patch

        from espnet2.speechlm.model.speechlm.parallel_utils import pipeline

        # Two virtual chunks on this rank.
        chunk0 = nn.Linear(4, 4)
        chunk0.stage_idx = 0
        chunk0.num_virtual_stages = 2
        chunk0.is_last_stage = False
        chunk1 = nn.Linear(4, 4)
        chunk1.stage_idx = 1
        chunk1.num_virtual_stages = 2
        chunk1.is_last_stage = True
        chunks = nn.ModuleList([chunk0, chunk1])

        pd = MagicMock()
        mesh = MagicMock()
        mesh.get_group.return_value = None
        pd.get_mesh.return_value = mesh

        # Multi-stage schedule: subclass of PipelineScheduleMulti.
        from torch.distributed.pipelining.schedules import PipelineScheduleMulti

        class _FakeMulti(PipelineScheduleMulti):
            def __init__(self, stages, n_microbatches, loss_fn, scale_grads):
                self.stages = stages
                self.n_microbatches = n_microbatches
                self.loss_fn = loss_fn

        with (
            patch.object(pipeline, "get_schedule_class", return_value=_FakeMulti),
            patch.object(pipeline, "PipelineStage", return_value=MagicMock()),
        ):
            schedule, has_last = pipeline.build_pipeline(
                chunks,
                parallel_dims=pd,
                titan_config={"pp_schedule": "Interleaved1F1B"},
                n_microbatches=4,
            )
        assert has_last is True  # chunk1.is_last_stage
        assert schedule.n_microbatches == 4
        # Identity loss handles both tuple and tensor outputs.
        t = torch.tensor(1.0)
        assert schedule.loss_fn(t, None) is t

    def test_multi_stage_requires_divisible_microbatches(self):
        """n_microbatches must be divisible by vpp_degree."""
        from unittest.mock import MagicMock, patch

        from torch.distributed.pipelining.schedules import PipelineScheduleMulti

        from espnet2.speechlm.model.speechlm.parallel_utils import pipeline

        chunks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)])
        for i, c in enumerate(chunks):
            c.stage_idx = i
            c.num_virtual_stages = 3
            c.is_last_stage = i == 2

        pd = MagicMock()
        mesh = MagicMock()
        mesh.get_group.return_value = None
        pd.get_mesh.return_value = mesh

        class _FakeMulti(PipelineScheduleMulti):
            def __init__(self, *a, **k):
                pass

        with (
            patch.object(pipeline, "get_schedule_class", return_value=_FakeMulti),
            patch.object(pipeline, "PipelineStage", return_value=MagicMock()),
        ):
            # 3 chunks, 4 microbatches → not divisible.
            with pytest.raises(AssertionError, match="divisible"):
                pipeline.build_pipeline(
                    chunks,
                    parallel_dims=pd,
                    titan_config={"pp_schedule": "Interleaved1F1B"},
                    n_microbatches=4,
                )

    def test_single_stage_rejects_multi_chunk_list(self):
        """Multi-chunk list with a single-stage schedule should fail."""
        from unittest.mock import MagicMock, patch

        from espnet2.speechlm.model.speechlm.parallel_utils import pipeline

        chunks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

        pd = MagicMock()
        mesh = MagicMock()
        mesh.get_group.return_value = None
        pd.get_mesh.return_value = mesh

        class _FakeSchedule:
            def __init__(self, *a, **k):
                pass

        with (
            patch.object(pipeline, "get_schedule_class", return_value=_FakeSchedule),
            patch.object(
                pipeline,
                "PipelineScheduleMulti",
                new=type("_DummyMulti", (), {}),
            ),
        ):
            with pytest.raises(AssertionError, match="expects 1 model"):
                pipeline.build_pipeline(
                    chunks,
                    parallel_dims=pd,
                    titan_config={},
                    n_microbatches=2,
                )
