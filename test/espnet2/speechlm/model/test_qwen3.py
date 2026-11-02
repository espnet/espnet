"""CPU tests for speechlm/parallel_utils/qwen3.py."""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torchtitan", reason="torchtitan not installed")
qwen3_moe = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")

from espnet2.speechlm.model.speechlm.parallel_utils.qwen3 import (  # noqa: E402
    _is_moe_layer,
    apply_activation_checkpoint_qwen3,
    apply_torch_compile_qwen3,
)

# Older TorchTitan compile helpers require this private PyTorch API.
_skip_no_set_lru_cache = pytest.mark.skipif(
    not hasattr(torch._C._dynamo.eval_frame, "_set_lru_cache"),
    reason="torch._C._dynamo.eval_frame._set_lru_cache unavailable on this torch",
)


def _make_moe_block(num_experts=4, hidden_size=8, intermediate_size=16, top_k=2):
    config = qwen3_moe.Qwen3MoeConfig(
        num_experts=num_experts,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        num_experts_per_tok=top_k,
    )
    return qwen3_moe.Qwen3MoeSparseMoeBlock(config)


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
        go through the else branch and get wrapped whole.
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        moe_layer = _MockMoeLayer()
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

        moe_layer = _MockMoeLayer()
        dense_layer = _MockDenseLayer()
        model = _MockHFModel([moe_layer, dense_layer])

        out = apply_activation_checkpoint_qwen3(model, ac_config=1.0, mode="moe")
        # MoE layer in "moe" mode: only .mlp is wrapped, layer itself is not.
        assert not isinstance(out.model.layers[0], CheckpointWrapper)
        assert isinstance(out.model.layers[0].mlp, CheckpointWrapper)
        # Dense layer in "moe" mode falls through to the else branch and is
        # wrapped at the layer level.
        assert isinstance(out.model.layers[1], CheckpointWrapper)


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


class TestParallelizeQwen3HF:
    def test_ac_only_no_fsdp_no_compile(self, monkeypatch):
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
        # and pp_enabled are False. parallelize_qwen3_hf
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
        monkeypatch.setattr(model, "cuda", lambda: model)

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
    def test_compile_branch(self, monkeypatch):
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
        monkeypatch.setattr(model, "cuda", lambda: model)

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
