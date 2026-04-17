# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Grouped MoE block for HuggingFace Qwen3 MoE models.

This module provides two MoE block implementations:

1. ``GroupedMoeBlock`` (EP=1): All experts on every rank. Uses
   ``torch._grouped_mm`` for fused multi-expert computation.

2. ``GroupedMoeBlockEP`` (EP>1): Expert Parallel. Each rank holds a
   subset of experts. Uses DeepEP for all-to-all token dispatch/combine
   and ``torch._grouped_mm`` for local expert computation.

Data flow (GroupedMoeBlock, EP=1):
    1. Route tokens (gate computation)
    2. Sort token-expert pairs by expert index
    3. Count tokens per expert (bincount)
    4. Process all experts via fused grouped_mm
    5. Permutation scatter + weighted sum

Data flow (GroupedMoeBlockEP, EP>1):
    1. Route tokens (gate computation, all experts visible)
    2. DeepEP dispatch: send tokens to the EP rank owning the expert
    3. Local expert computation via fused grouped_mm
    4. DeepEP combine: send results back to originating ranks
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torchtitan.distributed.deepep import (
    combine_tokens,
    dispatch_tokens,
    sync_combine,
)
from torchtitan.distributed.deepep.deepep import get_buffer, get_hidden_bytes
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
)

logger = logging.getLogger(__name__)


def _route(gate, hidden_states_flat, top_k, norm_topk_prob):
    """Top-k softmax routing shared by GroupedMoeBlock and GroupedMoeBlockEP."""
    router_logits = gate(hidden_states_flat)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states_flat.dtype)
    selected_experts = selected_experts.to(torch.int64)
    return router_logits, routing_weights, selected_experts


def _run_grouped_mm(w1, w2, w3, x, num_tokens_per_expert):
    """Core SwiGLU grouped_mm computation.

    SwiGLU: out = down_proj(silu(gate_proj(x)) * up_proj(x))
    Uses torch._grouped_mm which requires bfloat16 inputs.

    Weights are expected to be pre-transposed and contiguous:
        w1 (gate_proj): (num_experts, hidden_size, intermediate_size)
        w2 (down_proj): (num_experts, intermediate_size, hidden_size)
        w3 (up_proj):   (num_experts, hidden_size, intermediate_size)
    """
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    x_bf16 = x if x.dtype == torch.bfloat16 else x.bfloat16()
    w1 = w1 if w1.dtype == torch.bfloat16 else w1.bfloat16()
    w2 = w2 if w2.dtype == torch.bfloat16 else w2.bfloat16()
    w3 = w3 if w3.dtype == torch.bfloat16 else w3.bfloat16()
    h = F.silu(torch._grouped_mm(x_bf16, w1, offs=offsets))
    h = h * torch._grouped_mm(x_bf16, w3, offs=offsets)
    return torch._grouped_mm(h, w2, offs=offsets).type_as(x)


class GroupedExperts(nn.Module):
    """Stacked expert weights with fused grouped_mm computation.

    Stacks individual expert Linear weights into 3D Parameter tensors
    and uses torch._grouped_mm for fused multi-expert matrix multiplication.

    Weights are stored pre-transposed and contiguous so that
    torch._grouped_mm can use them directly without per-forward transposes.

    Stored weight layout (Qwen3 MoE SwiGLU, no biases):
        w1 (gate_proj): (num_experts, hidden_size, intermediate_size)
        w2 (down_proj): (num_experts, intermediate_size, hidden_size)
        w3 (up_proj):   (num_experts, hidden_size, intermediate_size)

    Forward: out = down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        experts: List of Qwen3MoeMLP modules to stack.
    """

    def __init__(self, experts: list):
        super().__init__()
        self.num_experts = len(experts)

        # Verify no biases (Qwen3 MoE experts don't have biases)
        assert (
            experts[0].gate_proj.bias is None
        ), "GroupedExperts does not support biases"

        # Preserve frozen state from source experts so that freeze_param
        # settings applied before grouped-MoE replacement are not lost.
        src_requires_grad = experts[0].gate_proj.weight.requires_grad

        # Stack weights from individual experts into 3D tensors,
        # pre-transposed and contiguous for torch._grouped_mm.
        # Original HF layout: (E, out_features, in_features)
        # Stored layout: (E, in_features, out_features) — ready for x @ w
        self.w1 = nn.Parameter(
            torch.stack([e.gate_proj.weight for e in experts])
            .transpose(-2, -1)
            .contiguous(),
            requires_grad=src_requires_grad,
        )
        self.w2 = nn.Parameter(
            torch.stack([e.down_proj.weight for e in experts])
            .transpose(-2, -1)
            .contiguous(),
            requires_grad=src_requires_grad,
        )
        self.w3 = nn.Parameter(
            torch.stack([e.up_proj.weight for e in experts])
            .transpose(-2, -1)
            .contiguous(),
            requires_grad=src_requires_grad,
        )

        self.hidden_size = self.w1.shape[1]
        self.intermediate_size = self.w1.shape[2]

        logger.info(
            f"GroupedExperts: stacked {self.num_experts} experts, "
            f"w1={list(self.w1.shape)}, w2={list(self.w2.shape)}, "
            f"w3={list(self.w3.shape)}"
        )

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, "
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Process tokens through stacked experts using grouped_mm.

        Args:
            x: Input tokens sorted by expert, shape (total_tokens, hidden_dim).
            num_tokens_per_expert: Token count per expert,
                shape (num_experts,).

        Returns:
            Output tensor, shape (total_tokens, hidden_dim).
        """
        # Extract local tensors from DTensor (torch._grouped_mm requires
        # regular tensors, not DTensors — needed for FSDP2 compatibility)
        if isinstance(self.w1, DTensor):
            w1, w2, w3 = self.w1.to_local(), self.w2.to_local(), self.w3.to_local()
        else:
            w1, w2, w3 = self.w1, self.w2, self.w3

        return _run_grouped_mm(w1, w2, w3, x, num_tokens_per_expert)


class GroupedMoeBlock(Qwen3MoeSparseMoeBlock):
    """MoE block using fused grouped_mm for all experts (no EP).

    Inherits from `Qwen3MoeSparseMoeBlock` so that HF's `OutputRecorder`
    (which uses `isinstance` check) can find this module and capture
    router_logits for the load balancing auxiliary loss.

    All experts reside on every rank. Uses GroupedExperts with
    torch._grouped_mm for fused multi-expert computation instead of a
    Python loop over individual experts.

    Args:
        original_block: The original Qwen3MoeSparseMoeBlock to replace.
    """

    def __init__(self, original_block: Qwen3MoeSparseMoeBlock):
        # Skip parent __init__ (which would create all experts from config).
        # Directly call nn.Module.__init__ instead.
        nn.Module.__init__(self)

        self.num_experts = original_block.num_experts
        self.top_k = original_block.top_k
        self.norm_topk_prob = original_block.norm_topk_prob

        # Router (replicated on all ranks — same gate weights)
        self.gate = original_block.gate

        # Stack ALL experts into GroupedExperts for fused grouped_mm
        self.experts = GroupedExperts(list(original_block.experts))

        logger.info(
            f"GroupedMoeBlock: {self.num_experts} experts, "
            f"top_k={self.top_k}, using grouped_mm"
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """Forward pass with fused grouped_mm computation.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Tuple of (output, router_logits):
                - output: shape (batch, seq_len, hidden_dim)
                - router_logits: shape (batch * seq_len, num_experts)
        """
        bsz, seq_len, hidden_dim = hidden_states.shape
        num_tokens = bsz * seq_len
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits, routing_weights, selected_experts = _route(
            self.gate,
            hidden_states_flat,
            self.top_k,
            self.norm_topk_prob,
        )

        # --- Step 2: Sort by expert index ---
        flat_expert_indices = selected_experts.view(-1)  # (T*K,)
        sort_order = flat_expert_indices.argsort(stable=True)  # (T*K,)

        # Gather tokens in expert-sorted order
        token_indices = (
            torch.arange(num_tokens, device=hidden_states.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )  # (T*K,)
        sorted_tokens = hidden_states_flat[token_indices[sort_order]]  # (T*K, D)

        # Count tokens per expert
        num_tokens_per_expert = torch.bincount(
            flat_expert_indices, minlength=self.num_experts
        )

        # --- Step 3: Process via grouped_mm (with padding for alignment) ---
        processed_tokens = self.experts(sorted_tokens, num_tokens_per_expert)

        # --- Step 4: Unsort via gather + weighted sum ---
        # Compute inverse permutation via cheap int-only scatter (2MB),
        # then gather the full (T*K, D) tensor (sequential writes, faster
        # than the scatter approach which has random writes on 1.8GB).
        TK = num_tokens * self.top_k
        inv_sort_order = torch.empty(
            TK, dtype=sort_order.dtype, device=sort_order.device
        )
        inv_sort_order[sort_order] = torch.arange(
            TK, dtype=sort_order.dtype, device=sort_order.device
        )
        unsorted = processed_tokens[inv_sort_order]  # gather: sequential writes
        # routing_weights is already (T, K) in original token order
        final_output = (
            unsorted.view(num_tokens, self.top_k, hidden_dim)
            * routing_weights.unsqueeze(-1)
        ).sum(dim=1)

        self._last_router_logits = router_logits
        return final_output.view(bsz, seq_len, hidden_dim), router_logits


class GroupedMoeBlockEP(Qwen3MoeSparseMoeBlock):
    """MoE block with Expert Parallelism via DeepEP.

    Each EP rank holds ``num_experts // ep_degree`` local experts.
    Token routing is computed on all ranks (full expert visibility),
    then DeepEP dispatches tokens to the rank owning the selected
    expert and combines results back.

    Inherits from ``Qwen3MoeSparseMoeBlock`` so that HF's
    ``OutputRecorder`` (which uses ``isinstance`` check) can find
    this module and capture router_logits for the load balancing loss.

    Args:
        original_block: The original Qwen3MoeSparseMoeBlock to replace.
        parallel_dims: TorchTitan ParallelDims with EP mesh built.
    """

    def __init__(self, original_block: Qwen3MoeSparseMoeBlock, parallel_dims):
        nn.Module.__init__(self)

        self.num_experts = original_block.num_experts
        self.top_k = original_block.top_k
        self.norm_topk_prob = original_block.norm_topk_prob

        ep_mesh = parallel_dims.get_mesh("ep")
        ep_degree = ep_mesh.size()
        assert self.num_experts % ep_degree == 0, (
            f"num_experts ({self.num_experts}) must be divisible by "
            f"ep_degree ({ep_degree})"
        )

        self.ep_group = ep_mesh.get_group()
        self.num_local_experts = self.num_experts // ep_degree
        self.ep_enabled = True  # survives checkpoint_wrapper via __getattr__

        self.gate = original_block.gate

        # Stack all experts, then shard on expert dim via DTensor.
        # distribute_tensor(Shard(0)) partitions the expert dimension across
        # EP ranks — each rank materializes only num_local_experts weights.
        self.experts = GroupedExperts(list(original_block.experts))
        for param_name, param in self.experts.named_parameters(recurse=False):
            self.experts.register_parameter(
                param_name,
                nn.Parameter(
                    distribute_tensor(param, ep_mesh, [Shard(0)]),
                    requires_grad=param.requires_grad,
                ),
            )

        # Eagerly initialize the DeepEP buffer so the first forward call
        # (which runs inside a torch.compiled region) doesn't pay the
        # one-time allocation cost.
        dummy = torch.empty(
            1,
            self.experts.hidden_size,
            dtype=torch.bfloat16,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        get_buffer(self.ep_group, get_hidden_bytes(dummy))

        # Workaround: EP>1 backward produces aten.index_add (via index_select
        # backward in DeepEP's _permute_tokens). Inductor has a conditional
        # decomp for index_add that returns NotImplemented for bf16, but its
        # mere presence in the decomp table conflicts with make_fallback on
        # newer PyTorch (e.g. required for Blackwell/sm_100). Removing it lets
        # Inductor use the ATen kernel directly as a fallback — same behavior
        # the bf16 decomp was already falling back to anyway.
        import torch._inductor.decomposition as _ind_decomp

        _ind_decomp.decompositions.pop(torch.ops.aten.index_add.default, None)

        logger.info(
            f"GroupedMoeBlockEP: {self.num_experts} total, "
            f"{self.num_local_experts} local (ep={ep_degree}), "
            f"top_k={self.top_k}"
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """Forward pass with DeepEP dispatch/combine.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Tuple of (output, router_logits):
                - output: shape (batch, seq_len, hidden_dim)
                - router_logits: shape (batch * seq_len, num_experts)
        """
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits, routing_weights, selected_experts = _route(
            self.gate,
            hidden_states_flat,
            self.top_k,
            self.norm_topk_prob,
        )

        # --- Step 2: DeepEP Dispatch ---
        dispatched_input, tokens_per_expert, dispatch_state = dispatch_tokens(
            hidden_states_flat,
            selected_experts,
            routing_weights,
            self.num_local_experts,
            self.num_experts,
            self.ep_group,
            score_before_experts=False,
        )

        # --- Step 3: Local expert computation ---
        expert_output = self.experts(dispatched_input, tokens_per_expert)

        # --- Step 4: DeepEP Combine ---
        final_output = combine_tokens(expert_output, dispatch_state)
        sync_combine()

        self._last_router_logits = router_logits
        return final_output.view(bsz, seq_len, hidden_dim), router_logits
