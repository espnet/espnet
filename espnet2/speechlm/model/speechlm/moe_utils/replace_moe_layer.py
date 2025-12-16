# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""DeepSpeed Expert Parallelism wrapper for Qwen3 MoE layers.

This module provides functionality to convert standard Qwen3 MoE layers
to DeepSpeed expert-parallel versions for distributed training across
multiple GPUs with expert sharding.
"""

import copy
from functools import partial
from typing import Tuple

import torch
import torch.nn.functional as F
from deepspeed import comm as dist
from deepspeed.moe.layer import MoE as DeepSpeed_MoE
from deepspeed.moe.sharded_moe import _AllToAll
from deepspeed.utils import groups
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
)


class Qwen3MoeSparseMoeBlock_DeepSpeed_EP(DeepSpeed_MoE):
    """DeepSpeed expert-parallel wrapper for Qwen3 MoE blocks.

    This class wraps a Qwen3MoeSparseMoeBlock to enable expert parallelism
    using DeepSpeed's distributed training infrastructure. Experts are
    sharded across multiple GPUs to reduce memory requirements.

    Args:
        module: Original Qwen3MoeSparseMoeBlock to parallelize
        ep_size: Expert parallelism size (number of processes)

    Attributes:
        num_local_experts: Number of experts on this process
        ep_rank: Rank within the expert parallel group
        ep_group: DeepSpeed expert parallel process group
    """

    def __init__(self, module: torch.nn.Module, ep_size: int) -> None:
        """Initialize expert-parallel MoE layer.

        Args:
            module: Original MoE module to wrap
            ep_size: Number of processes for expert parallelism
        """
        # NOTE: We only initialize as torch.nn.Module, not DeepSpeed_MoE
        # to avoid conflicts with DeepSpeed's initialization
        torch.nn.Module.__init__(self)

        # Internal configuration
        self.enable_expert_tensor_parallelism = False
        self.num_experts = len(module.experts)
        self.ep_size = ep_size
        self.num_local_experts = self.num_experts // self.ep_size
        self.norm_topk_prob = module.norm_topk_prob
        self.top_k = module.top_k

        # Validate configuration
        if self.ep_size <= 1:
            raise ValueError(f"ep_size must be > 1, got {ep_size}")
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible "
                f"by ep_size ({ep_size})"
            )

        # (3) setup the ep_group
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.set_deepspeed_parallelism()

        # (4) copy the expert modules from the original module
        self.gate = copy.deepcopy(module.gate)
        start = self.ep_rank * self.num_local_experts
        end = (self.ep_rank + 1) * self.num_local_experts
        self.experts = torch.nn.ModuleList(
            [copy.deepcopy(e) for e in module.experts[start:end]]
        )
        for expert in self.experts:
            for param in expert.parameters():
                param.allreduce = False
                param.group_name = self.expert_group_name

    def set_deepspeed_parallelism(
        self, use_data_before_expert_parallel_: bool = False
    ) -> None:
        """Set up DeepSpeed parallelism groups.

        Args:
            use_data_before_expert_parallel_: Whether to use data parallelism
                before expert parallelism in group creation
        """
        self._create_process_groups(
            use_data_before_expert_parallel_=(use_data_before_expert_parallel_)
        )

    def _create_process_groups(
        self, use_data_before_expert_parallel_: bool = False
    ) -> None:
        # Create process group for a layer if needed
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism):
                # Condition 1 - no groups.mpu means no tensor parallelism
                # Condition 2 - disabling expert tensor parallelism on purpose
                groups._create_expert_and_data_parallel(
                    self.ep_size,
                    use_data_before_expert_parallel_=use_data_before_expert_parallel_,
                )
            else:
                # expert tensor parallelism is enabled
                groups._create_expert_data_and_model_parallel(
                    self.ep_size,
                    mpu=groups.mpu,
                    use_data_before_expert_parallel_=use_data_before_expert_parallel_,
                )
        # Set the group handle for the MOELayer (deepspeed_moe) object
        self.ep_group = groups._get_expert_parallel_group(self.expert_group_name)
        self.ep_rank = dist.get_rank(group=self.ep_group)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through expert-parallel MoE layer.

        Performs routing, expert computation with all-to-all communication,
        and combines expert outputs. Experts are distributed across processes
        and communication happens via DeepSpeed all-to-all operations.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_dim]

        Returns:
            Tuple of:
                - Output hidden states [batch, seq_len, hidden_dim]
                - Router logits for auxiliary loss computation
        """
        batch_size, sequence_length, hidden_dim = hidden_states.size()
        num_token = batch_size * sequence_length
        hidden_states = hidden_states.view(num_token, hidden_dim)

        # (1) router forward and dispatch
        router_logits = self.gate(hidden_states)
        location_one_hot, combine_weights = self.prepare_dispatch(router_logits)

        # (2) hidden_states exchange and forward
        expert_input = torch.matmul(
            location_one_hot.type_as(hidden_states), hidden_states
        )
        expert_input = _AllToAll.apply(self.ep_group, expert_input)

        expert_input = expert_input.reshape(
            self.ep_size, self.num_local_experts, -1, hidden_dim
        )
        expert_input = expert_input.chunk(self.num_local_experts, dim=1)
        expert_output = torch.stack(
            [e(h) for e, h in zip(self.experts, expert_input)], dim=1
        )
        expert_output = expert_output.reshape(
            self.ep_size * self.num_local_experts, -1, hidden_dim
        )
        expert_output = _AllToAll.apply(self.ep_group, expert_output)

        # (3) recover the hidden states
        hidden_states = torch.matmul(
            combine_weights.reshape(num_token, -1).type_as(hidden_states),
            expert_output.reshape(-1, hidden_dim),
        )
        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_logits

    @torch.no_grad()
    def prepare_dispatch(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare dispatch tensors for expert routing.

        Computes top-k routing, creates one-hot dispatch tensors, and
        calculates combine weights. All operations are performed in FP32
        for numerical stability.

        Args:
            logits: Router output logits [num_tokens, num_experts]

        Returns:
            Tuple of:
                - location_one_hot: Dispatch tensor [num_experts, capacity, num_tokens]
                - combine_weights: Weight tensor [num_tokens, num_experts, capacity]
        """

        # (1) softmax and top_k
        prob = F.softmax(logits, dim=1, dtype=torch.float)
        topk_weights, topk_index = torch.topk(prob, self.top_k, dim=1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

        # (2) bool and weight mask
        masked_weights = torch.zeros_like(prob).scatter_(1, topk_index, topk_weights)
        masked_bool = masked_weights.bool()

        # (3) find capacity
        capacity = torch.max(torch.sum(masked_bool, dim=0))
        dist.all_reduce(capacity, op=dist.ReduceOp.MAX, group=self.ep_group)
        # Keep position 0 as meaningless in later one-hot op
        capacity += 1

        # (4) one-hot location
        location = torch.cumsum(masked_bool, dim=0)
        location = location * masked_bool
        location_one_hot = F.one_hot(location, capacity).float()
        combine_weights = masked_weights.unsqueeze(2) * location_one_hot
        location_one_hot = location_one_hot.permute(1, 2, 0)

        # Returns: [num_experts, capacity, num_tokens],
        #          [num_token, num_experts, capacity]
        return location_one_hot, combine_weights


def replace_moe_layer(
    model: torch.nn.Module,
    ep_size: int,
    original_cls: type,
    ep_cls: type,
) -> torch.nn.Module:
    """Replace MoE layers with expert-parallel versions.

    Recursively traverses the model and replaces all instances of
    original_cls with ep_cls initialized with expert parallelism.

    Args:
        model: Model to modify
        ep_size: Expert parallelism size
        original_cls: Original MoE layer class to replace
        ep_cls: Expert-parallel class to use as replacement

    Returns:
        Modified model with expert-parallel MoE layers
    """
    if ep_size <= 1:
        return model

    def recursive_replace(module, parent_name=""):
        """Recursively replace MoE layers in the module tree."""
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, original_cls):
                new_child = ep_cls(child, ep_size)
                setattr(module, name, new_child)
                # del child
            else:
                recursive_replace(child, full_name)

    recursive_replace(model)
    return model


# Convenience function for replacing Qwen3 MoE layers
replace_qwen3_moe_layer = partial(
    replace_moe_layer,
    original_cls=Qwen3MoeSparseMoeBlock,
    ep_cls=Qwen3MoeSparseMoeBlock_DeepSpeed_EP,
)

__all__ = [
    "replace_qwen3_moe_layer",
]
