# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parallelization utilities for SpeechLM models.

This module provides model-specific parallelization strategies for different
model architectures. Each strategy applies FSDP2 wrapping optimized for
that model structure.

Usage:
    from espnet2.speechlm.model.speechlm.parallel_utils import (
        init_parallel_dims,
        parallel_strategies,
    )

    # Initialize parallel dimensions
    parallel_dims, local_rank, global_rank = init_parallel_dims(titan_config)

    # Get the parallelization function for a specific model
    parallelize_fn = parallel_strategies["qwen3"]
    model = parallelize_fn(model, parallel_dims, titan_config)
"""

from espnet2.speechlm.model.speechlm.parallel_utils.parallel_dims import (
    init_parallel_dims,
)
from espnet2.speechlm.model.speechlm.parallel_utils.pipeline import (
    build_pipeline,
)
from espnet2.speechlm.model.speechlm.parallel_utils.qwen3 import (
    parallelize_qwen3_hf,
)

# Registry of parallelization strategies for different model series
# Each function has signature: (model, parallel_dims, titan_config) -> model
parallel_strategies = {
    "qwen3": parallelize_qwen3_hf,
}

__all__ = [
    "init_parallel_dims",
    "build_pipeline",
    "parallel_strategies",
]
