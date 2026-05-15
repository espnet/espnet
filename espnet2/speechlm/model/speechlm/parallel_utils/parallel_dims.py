# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parallel dimensions initialization utilities.

This module provides functions for initializing TorchTitan's ParallelDims
for FSDP2, pipeline parallel, and expert parallel training.

Note: This module assumes torch.distributed is already initialized and
CUDA device is already set before calling init_parallel_dims().
"""

import logging
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
from torchtitan.distributed import ParallelDims

logger = logging.getLogger(__name__)


def init_parallel_dims(
    titan_config: Dict[str, Any],
) -> Tuple[ParallelDims, int, int]:
    """Create ParallelDims for distributed training.

    Supports FSDP2 (dp_shard), HSDP (dp_replicate), pipeline parallelism
    (pp), and expert parallelism (ep).

    The constraint ``dp_replicate * dp_shard * pp == world_size`` is
    enforced by TorchTitan; ``dp_shard=-1`` auto-computes the remainder.

    EP borrows from the FSDP dimension — it does NOT consume additional
    world_size. TorchTitan internally computes ``efsdp = dp_shard / ep``
    for the expert FSDP mesh. For example, with 8 GPUs and ep=8:
    dense params use fsdp=8, expert params use efsdp=1 + ep=8.

    This function assumes:
    - torch.distributed is already initialized (via dist.init_process_group)
    - CUDA device is already set (via torch.cuda.set_device)

    Args:
        titan_config: TorchTitan configuration dictionary containing:
            - dp_replicate: HSDP replicate degree (default: 1)
            - dp_shard: FSDP sharding degree (-1 = auto, default: -1)
            - pp_degree: Pipeline parallel degree (default: 1)
            - ep: Expert parallel degree (default: 1). Must divide
              dp_shard evenly.

    Returns:
        Tuple of (parallel_dims, local_rank, global_rank):
            - parallel_dims: ParallelDims object with device meshes built
            - local_rank: Local rank within the node (current CUDA device)
            - global_rank: Global rank across all nodes
    """
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    local_rank = torch.cuda.current_device()

    ep = titan_config.get("ep", 1)

    parallel_dims = ParallelDims(
        dp_replicate=titan_config.get("dp_replicate", 1),
        dp_shard=titan_config.get("dp_shard", -1),  # -1 = auto
        cp=1,
        tp=1,
        pp=titan_config.get("pp_degree", 1),
        ep=ep,
        etp=1,
        world_size=world_size,
    )

    parallel_dims.build_mesh()

    ep_info = ""
    if ep > 1:
        efsdp = parallel_dims.dp_shard // ep
        ep_info = f", ep={ep}, efsdp={efsdp}"

    logger.info(
        f"Built device mesh: world_size={world_size}, "
        f"dp_replicate={parallel_dims.dp_replicate}, "
        f"dp_shard={parallel_dims.dp_shard}, "
        f"pp={parallel_dims.pp}{ep_info}"
    )

    return parallel_dims, local_rank, global_rank
