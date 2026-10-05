# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parallelization utilities for HuggingFace Qwen3 models.

This module provides activation checkpointing, torch.compile, and FSDP2
wrapping for HuggingFace Qwen3 (dense and MoE)
models used in the SpeechLM framework. It follows TorchTitan's
parallelization patterns adapted for the HuggingFace model structure.

HuggingFace Qwen3 model structure:
    model.model.embed_tokens  - Token embeddings
    model.model.layers        - List of transformer layers
    model.model.norm          - Final RMSNorm
    model.lm_head             - Output projection

For MoE models (e.g., Qwen3-30B-A3B), some layers have:
    layer.mlp = Qwen3MoeSparseMoeBlock
        .gate: Qwen3MoeTopKRouter   # Router
        .experts: Qwen3MoeExperts  # Native Transformers grouped MM experts

Additional multimodal components (added by ParallelHFModel):
    model.multimodal_io_dict  - Dict of multimodal IO handlers
    model.adaptor             - Dict of linear adaptors for continuous modalities
    model.stream_emb          - Stream embeddings
"""

import logging
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torchtitan.distributed import ParallelDims

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

    Order: AC -> torch.compile -> FSDP
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

    # 1. Activation Checkpointing
    ac_config = titan_config.get("activation_checkpoint", 0.0)
    ac_mode = titan_config.get("activation_checkpoint_mode", "full")
    model = apply_activation_checkpoint_qwen3(
        model,
        ac_config=ac_config,
        mode=ac_mode,
        vpp_index=vpp_index,
    )

    # 2. Torch Compile
    if titan_config.get("compile", False):
        model = apply_torch_compile_qwen3(model, titan_config)

    # 3. FSDP
    if parallel_dims.fsdp_enabled:
        model = apply_fsdp_qwen3(model, parallel_dims, titan_config)
    else:
        model = model.cuda()

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

    # (1) Build FSDP config
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

    def _move_and_shard(module: nn.Module):
        """Move module to GPU and immediately shard via FSDP."""
        module.to(device)
        fully_shard(module, **fsdp_config)

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
        _move_and_shard(layer)

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
    _setup_fsdp_prefetch(model)

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
            if mode in ["moe", "moe_and_full"] and _is_moe_layer(
                model.model.layers[idx]
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


def _setup_fsdp_prefetch(model: nn.Module) -> None:
    """Set up 1-layer-ahead FSDP prefetch for forward and backward passes.

    After each layer's all-gather copy-out, FSDP will immediately issue
    the all-gather for the next layer, overlapping communication with
    the current layer's compute.

    Tolerates pruned PP stage models where embed_tokens, norm, lm_head,
    or stream_emb may be None.

    Memory cost: ~1.2GB for one prefetched MoE layer (unsharded bf16 params).

    Args:
        model: FSDP-wrapped HuggingFace Qwen3 model (full or PP stage)
    """

    layers = [
        layer for layer in model.model.layers if not isinstance(layer, nn.Identity)
    ]
    num_layers = len(layers)
    if num_layers == 0:
        return

    # Forward: each module prefetches the next one in execution order
    if model.model.embed_tokens is not None:
        model.model.embed_tokens.set_modules_to_forward_prefetch([layers[0]])
    for i in range(num_layers - 1):
        layers[i].set_modules_to_forward_prefetch([layers[i + 1]])
    last_layer_fwd_prefetch = [
        m
        for m in [model.model.norm, model.lm_head, getattr(model, "stream_emb", None)]
        if m is not None
    ]
    if last_layer_fwd_prefetch:
        layers[-1].set_modules_to_forward_prefetch(last_layer_fwd_prefetch)

    # Backward: layers execute in reverse; each prefetches the previous one
    if model.lm_head is not None:
        model.lm_head.set_modules_to_backward_prefetch([layers[-1]])
    if getattr(model, "stream_emb", None) is not None:
        model.stream_emb.set_modules_to_backward_prefetch([layers[-1]])
    for i in range(num_layers - 1, 0, -1):
        layers[i].set_modules_to_backward_prefetch([layers[i - 1]])
    if model.model.embed_tokens is not None:
        layers[0].set_modules_to_backward_prefetch([model.model.embed_tokens])

    logger.info(f"Set up 1-layer FSDP prefetch on {num_layers} layers")
