# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parallelization utilities for HuggingFace Qwen3 models.

This module provides grouped MoE replacement, activation checkpointing,
torch.compile, and FSDP2 wrapping for HuggingFace Qwen3 (dense and MoE)
models used in the SpeechLM framework. It follows TorchTitan's
parallelization patterns adapted for the HuggingFace model structure.

HuggingFace Qwen3 model structure:
    model.model.embed_tokens  - Token embeddings
    model.model.layers        - List of transformer layers
    model.model.norm          - Final RMSNorm
    model.lm_head             - Output projection

For MoE models (e.g., Qwen3-30B-A3B), some layers have:
    layer.mlp = Qwen3MoeSparseMoeBlock
        .gate: nn.Linear(hidden_size, num_experts)   # Router
        .experts: nn.ModuleList of Qwen3MoeMLP        # Individual experts

Additional multimodal components (added by ParallelHFModel):
    model.multimodal_io_dict  - Dict of multimodal IO handlers
    model.adaptor             - Dict of linear adaptors for continuous modalities
    model.stream_emb          - Stream embeddings
"""

import logging
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torchtitan.distributed import ParallelDims

from espnet2.speechlm.model.speechlm.parallel_utils.grouped_moe import (
    GroupedMoeBlock,
    GroupedMoeBlockEP,
)

logger = logging.getLogger(__name__)


def _is_moe_layer(layer: nn.Module) -> bool:
    """Check if a transformer layer uses MoE (has gate + experts in mlp)."""
    return (
        hasattr(layer, "mlp")
        and hasattr(layer.mlp, "gate")
        and hasattr(layer.mlp, "experts")
    )


def parallelize_qwen3_hf(
    model: nn.Module,
    parallel_dims: ParallelDims,
    titan_config: Dict[str, Any],
    vpp_index: int = 0,
) -> nn.Module:
    """Apply parallelization to HuggingFace Qwen3 model.

    Order: Grouped MoE -> AC -> torch.compile -> FSDP
    (following TorchTitan's convention)

    Args:
        model: HuggingFace Qwen3 model (possibly wrapped with multimodal components)
        parallel_dims: TorchTitan ParallelDims object with device meshes
        titan_config: Configuration dict containing:
            - activation_checkpoint: AC ratio 0.0-1.0 (default: 0.0).
              1.0 = all layers, 0.5 = every other layer.
            - compile: Whether to enable torch.compile (default: false)
            - compile_mode: Compile mode (default: "default")
            - mixed_precision_param: Parameter dtype (default: "bfloat16")
            - mixed_precision_reduce: Reduce dtype (default: "float32")
            - reshard_after_forward: Whether to reshard params after forward
              (default: true). true saves memory, false is faster.

    Returns:
        Parallelized model
    """

    # 1. Grouped MoE (must come first — replaces MoE blocks with fused grouped_mm)
    model = apply_grouped_moe_qwen3(model, parallel_dims)

    # 2. Activation Checkpointing
    ac_config = titan_config.get("activation_checkpoint", 0.0)
    ac_mode = titan_config.get("activation_checkpoint_mode", "full")
    model = apply_activation_checkpoint_qwen3(
        model,
        ac_config=ac_config,
        mode=ac_mode,
        vpp_index=vpp_index,
    )

    # 3. Torch Compile
    if titan_config.get("compile", False):
        model = apply_torch_compile_qwen3(model, titan_config)

    # 4. FSDP
    if parallel_dims.fsdp_enabled:
        model = apply_fsdp_qwen3(model, parallel_dims, titan_config)

    return model


def memory_efficient_load_balancing_loss(
    gate_logits,
    num_experts=None,
    top_k=2,
):
    """Memory-efficient load balancing loss — numerically identical to HF version.

    Eliminates the massive one_hot tensor (N_total, K, E) by using bincount.
    Processes per-layer to avoid concatenating all router logits.

    Memory: O(E) instead of O(N_total * K * E).
    """
    if (
        gate_logits is None
        or not isinstance(gate_logits, tuple)
        or len(gate_logits) == 0
    ):
        return torch.tensor(0.0)

    device = gate_logits[0].device
    total_tokens = 0
    expert_counts = torch.zeros(num_experts, device=device)
    router_prob_sum = torch.zeros(num_experts, device=device)

    for layer_gate in gate_logits:
        total_tokens += layer_gate.shape[0]

        routing_weights = F.softmax(layer_gate, dim=-1, dtype=torch.float)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

        expert_counts = (
            expert_counts
            + torch.bincount(
                selected_experts.reshape(-1), minlength=num_experts
            ).float()
        )
        router_prob_sum = router_prob_sum + routing_weights.sum(dim=0)

    tokens_per_expert = expert_counts / total_tokens
    router_prob_per_expert = router_prob_sum / total_tokens

    return torch.dot(tokens_per_expert, router_prob_per_expert) * num_experts


def apply_grouped_moe_qwen3(
    model: nn.Module,
    parallel_dims: ParallelDims = None,
) -> nn.Module:
    """Replace MoE blocks with GroupedMoeBlock (EP=1) or GroupedMoeBlockEP (EP>1).

    Iterates through transformer layers, detects MoE layers, and replaces
    each Qwen3MoeSparseMoeBlock with a fused grouped_mm implementation.

    When ``parallel_dims.ep > 1``, uses ``GroupedMoeBlockEP`` which stacks
    all experts (sharded later via distribute_tensor) and uses DeepEP for
    token dispatch/combine.

    Must be applied BEFORE activation checkpointing, compile, and FSDP.

    Args:
        model: HuggingFace Qwen3 MoE model
        parallel_dims: TorchTitan ParallelDims. When ep > 1, the EP mesh
            is used to partition experts across ranks.

    Returns:
        Model with MoE blocks replaced
    """
    ep_enabled = parallel_dims is not None and parallel_dims.ep_enabled

    has_moe = False
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer, nn.Identity):
            continue
        layer = layer.cuda()
        if _is_moe_layer(layer) and not isinstance(
            layer.mlp, (GroupedMoeBlock, GroupedMoeBlockEP)
        ):
            if ep_enabled:
                layer.mlp = GroupedMoeBlockEP(layer.mlp, parallel_dims)
            else:
                layer.mlp = GroupedMoeBlock(layer.mlp)
            has_moe = True

    if has_moe:
        model.load_balancing_loss_func = memory_efficient_load_balancing_loss

    return model


def apply_fsdp_qwen3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    titan_config: Dict[str, Any],
) -> nn.Module:
    """Apply FSDP2 to HuggingFace Qwen3 model structure.

    Moves modules from CPU to GPU one FSDP unit at a time, then shards
    immediately. This avoids materializing the full model on every GPU
    (which would waste ~60GB for a 30B model). Peak GPU memory during
    init is ~1 transformer layer instead of the entire model.

    When EP is enabled (``parallel_dims.ep > 1``), MoE expert parameters
    are wrapped with a separate ``edp_mesh`` (expert data-parallel mesh)
    before the full layer is wrapped with the dense ``dp_mesh``. This
    ensures expert gradients are only reduced within the efsdp dimension,
    not across EP ranks (which hold different experts).

    Tolerates pruned PP stage models where some modules (embed_tokens,
    lm_head, norm, stream_emb, multimodal_io_dict, adaptor) may be None.

    Args:
        model: HuggingFace Qwen3 model (on CPU or GPU) to wrap with FSDP.
            May be a full model or a pruned PP stage.
        parallel_dims: TorchTitan ParallelDims with device meshes
        titan_config: Configuration dict

    Returns:
        FSDP-wrapped model (on GPU, sharded)
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # (1) Build FSDP config for dense parameters
    param_dtype = getattr(torch, titan_config.get("mixed_precision_param", "bfloat16"))
    reduce_dtype = getattr(torch, titan_config.get("mixed_precision_reduce", "float32"))
    pp_enabled = parallel_dims.pp_enabled
    reshard_after_forward = titan_config.get("reshard_after_forward", not pp_enabled)
    assert not (pp_enabled and reshard_after_forward), (
        "reshard_after_forward must be False when pipeline parallelism is enabled. "
        "Set reshard_after_forward: false in titan_config."
    )

    if parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
    else:
        dp_mesh = parallel_dims.get_mesh("fsdp")

    fsdp_config = {
        "mesh": dp_mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        ),
        "reshard_after_forward": reshard_after_forward,
    }

    # (1b) Build FSDP config for expert parameters (EP > 1)
    ep_degree = parallel_dims.ep
    ep_enabled = parallel_dims.ep_enabled
    edp_mesh = None
    fsdp_ep_config = None

    if ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        fsdp_ep_config = {
            "mesh": edp_mesh,
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype
            ),
            "reshard_after_forward": reshard_after_forward,
        }
        logger.info(
            f"EP FSDP: edp_mesh={edp_mesh_names}, ep_degree={ep_degree}, "
            f"efsdp_size={edp_mesh['efsdp'].size() if edp_mesh is not None else 'N/A'}"
        )

    def _move_and_shard(module: nn.Module):
        """Move module to GPU and immediately shard via FSDP."""
        module.to(device)
        fully_shard(module, **fsdp_config)

    def _move_and_shard_layer(layer: nn.Module):
        """Move a transformer layer to GPU and shard.

        When EP is enabled and the layer is MoE, wraps the
        already-sharded experts with FSDP on edp_mesh before wrapping
        the full layer on dp_mesh.
        """
        layer.to(device)

        if ep_enabled and getattr(layer.mlp, "ep_enabled", False):
            efsdp_size = edp_mesh["efsdp"].size()
            assert efsdp_size == 1, "FSDP on experts is not supported yet."
            fully_shard(layer.mlp.experts, **fsdp_ep_config)

        fully_shard(layer, **fsdp_config)

    # (2.1) input embeddings
    has_embed = model.model.embed_tokens is not None
    has_lm_head = model.lm_head is not None
    tied = (
        has_embed
        and has_lm_head
        and model.lm_head.weight is model.model.embed_tokens.weight
    )

    if tied:
        model.model.embed_tokens.to(device)
        model.lm_head.to(device)
        fully_shard([model.model.embed_tokens, model.lm_head], **fsdp_config)
        logger.info("Tied embed_tokens + lm_head: wrapped in single FSDP unit")
    else:
        if has_embed:
            _move_and_shard(model.model.embed_tokens)

    # (2.2) layers — move one at a time to avoid full-model GPU peak
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer, nn.Identity):
            continue
        _move_and_shard_layer(layer)

    # (2.3) norm, lm_head (if untied), stream_emb
    if model.model.norm is not None:
        _move_and_shard(model.model.norm)
    if has_lm_head and not tied:
        _move_and_shard(model.lm_head)
    if getattr(model, "stream_emb", None) is not None:
        _move_and_shard(model.stream_emb)

    # (2.4) root — moves remaining modules (multimodal_io_dict, adaptor, etc.)
    # NOTE(Jinchuan): The FSDP2 DTensor operation doesn't support convolution ops.
    # We put all remained peripheral modules to the root FSDP2 unit, where the conv
    # ops can always stay locally and will not trigger the DTensor check.
    model.to(device)
    fully_shard(model, **fsdp_config)

    logger.info(
        f"Incremental FSDP init complete — peak GPU memory: "
        f"{torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB"
    )

    # (2.5) Multi-layer FSDP prefetch (must be after all modules are sharded)
    _setup_fsdp_prefetch(model, titan_config, ep_enabled)

    return model


def apply_activation_checkpoint_qwen3(
    model: nn.Module,
    ac_config: Union[float, List[float]] = 0.0,
    mode: str = "full",
    vpp_index: int = 0,
) -> nn.Module:
    """Apply activation checkpointing to transformer layers.

    Wraps transformer layers with checkpoint_wrapper for memory savings.
    Must be applied before torch.compile and FSDP.

    When ``ac_config`` is a list, it specifies a per-virtual-stage ratio
    and ``vpp_index`` selects which entry to use. When it is a scalar,
    the same ratio applies to all stages.

    nn.Identity placeholder layers (from PP pruning) are skipped.

    Args:
        model: HuggingFace Qwen3 model (possibly PP-pruned with
            nn.Identity placeholders for non-local layers).
        ac_config: AC ratio (0.0-1.0) or list of per-virtual-stage
            ratios. 1.0 = all layers, 0.5 = every other layer, etc.
        mode: Checkpointing granularity:
            - "full": wrap the entire transformer layer (default)
            - "moe": wrap only layer.mlp on MoE layers, skip dense layers.
        vpp_index: Virtual pipeline stage index on this rank. Used as
            fallback when ``model.stage_idx`` is not set. When
            ``ac_config`` is a list, it is indexed by
            ``model.stage_idx`` (the global virtual stage index).

    Returns:
        Model with activation checkpointing applied
    """
    if isinstance(ac_config, list):
        stage_idx = getattr(model, "stage_idx")
        ratio = ac_config[stage_idx]
    else:
        ratio = ac_config

    if ratio <= 0.0:
        return model

    real_layers = [
        idx
        for idx, layer in enumerate(model.model.layers)
        if not isinstance(layer, nn.Identity)
    ]
    num_real = len(real_layers)
    num_to_checkpoint = max(1, round(num_real * ratio))

    count = 0
    for pos, idx in enumerate(real_layers):
        if (
            count < num_to_checkpoint
            and (pos + 1) * num_to_checkpoint > count * num_real
        ):
            if mode in ["moe", "moe_and_full"] and isinstance(
                model.model.layers[idx].mlp, (GroupedMoeBlock, GroupedMoeBlockEP)
            ):
                model.model.layers[idx].mlp = checkpoint_wrapper(
                    model.model.layers[idx].mlp
                )
            else:
                model.model.layers[idx] = checkpoint_wrapper(model.model.layers[idx])
            count += 1
        # NOTE(Jinchuan) this mode means: the selected layer only do AC on MoE; other
        # layers apply AC in the whole layers.
        elif mode == "moe_and_full":
            model.model.layers[idx] = checkpoint_wrapper(model.model.layers[idx])

    if hasattr(model.model, "norm"):
        model.model.norm = checkpoint_wrapper(model.model.norm)
    if hasattr(model.model, "embed_tokens") and model.model.embed_tokens is not None:
        model.model.embed_tokens = checkpoint_wrapper(model.model.embed_tokens)

    logger.info(
        f"Applied activation checkpointing to {count}/{num_real} real layers "
        f"(ratio={ratio}, mode={mode}, "
        f"stage_idx={getattr(model, 'stage_idx', vpp_index)})"
    )
    return model


def apply_torch_compile_qwen3(
    model: nn.Module,
    titan_config: Dict[str, Any],
) -> nn.Module:
    """Apply torch.compile to transformer layers.

    Compiles each transformer layer individually. Must be applied after
    activation checkpointing and before FSDP.

    Args:
        model: HuggingFace Qwen3 model
        titan_config: Configuration dict

    Returns:
        Model with compiled transformer layers
    """
    compile_mode = titan_config.get("compile_mode", "default")

    torch._dynamo.config.capture_scalar_outputs = True
    torch._C._dynamo.eval_frame._set_lru_cache(False)

    count = 0
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer, nn.Identity):
            continue
        model.model.layers[idx] = torch.compile(layer, mode=compile_mode)
        count += 1

    logger.info(
        f"Applied torch.compile (mode={compile_mode}) to "
        f"{count}/{len(model.model.layers)} layers"
    )

    return model


def _setup_fsdp_prefetch(
    model: nn.Module,
    titan_config: Dict[str, Any],
    ep_enabled: bool = False,
) -> None:
    """Set up 1-layer-ahead FSDP prefetch for forward and backward passes.

    After each layer's all-gather copy-out, FSDP will immediately issue
    the all-gather for the next layer, overlapping communication with
    the current layer's compute.

    When EP is enabled, explicit prefetch is always activated because
    EP's device-to-host syncs (for token count exchange) can interfere
    with FSDP's implicit prefetching. Additionally, MoE layers have an
    inner FSDP unit for experts (on edp_mesh) that is NOT automatically
    prefetched when the outer layer is prefetched. We explicitly include
    ``layer.mlp.experts`` in the prefetch list for MoE layers.

    Tolerates pruned PP stage models where embed_tokens, norm, lm_head,
    or stream_emb may be None.

    Memory cost: ~1.2GB for one prefetched MoE layer (unsharded bf16 params).

    Args:
        model: FSDP-wrapped HuggingFace Qwen3 model (full or PP stage)
        titan_config: Set ``prefetch`` to true to enable (default: false).
        ep_enabled: When True, forces prefetch on regardless of config.
    """

    layers = [
        layer for layer in model.model.layers if not isinstance(layer, nn.Identity)
    ]
    num_layers = len(layers)
    if num_layers == 0:
        return

    def _prefetch_targets(layer):
        """Return prefetch targets for a layer.

        For MoE-EP layers, includes both the layer itself and its inner
        experts FSDP unit (which lives on a separate edp_mesh). The
        inner experts unit is not automatically prefetched by FSDP when
        the outer layer is prefetched, so we must list it explicitly.
        """
        targets = [layer]
        if ep_enabled and hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            if getattr(layer.mlp, "ep_enabled", False):
                targets.append(layer.mlp.experts)
        return targets

    # Forward: each module prefetches the next one in execution order
    if model.model.embed_tokens is not None:
        model.model.embed_tokens.set_modules_to_forward_prefetch(
            _prefetch_targets(layers[0])
        )
    for i in range(num_layers - 1):
        layers[i].set_modules_to_forward_prefetch(_prefetch_targets(layers[i + 1]))
    last_layer_fwd_prefetch = [
        m
        for m in [model.model.norm, model.lm_head, getattr(model, "stream_emb", None)]
        if m is not None
    ]
    if last_layer_fwd_prefetch:
        layers[-1].set_modules_to_forward_prefetch(last_layer_fwd_prefetch)

    # Backward: layers execute in reverse; each prefetches the previous one
    if model.lm_head is not None:
        model.lm_head.set_modules_to_backward_prefetch(_prefetch_targets(layers[-1]))
    if getattr(model, "stream_emb", None) is not None:
        model.stream_emb.set_modules_to_backward_prefetch(_prefetch_targets(layers[-1]))
    for i in range(num_layers - 1, 0, -1):
        layers[i].set_modules_to_backward_prefetch(_prefetch_targets(layers[i - 1]))
    if model.model.embed_tokens is not None:
        layers[0].set_modules_to_backward_prefetch([model.model.embed_tokens])

    logger.info(
        f"Set up 1-layer FSDP prefetch on {num_layers} layers"
        f"{' (with EP expert prefetch)' if ep_enabled else ''}"
    )
