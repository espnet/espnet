# Copyright 2026 Carnegie Mellon University
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""Optional multi-GPU check of SpeechLM gradients using the real Liger kernel."""

import math
from test.espnet2.speechlm.model.test_parallel_fsdp import _check_gradients_and_updates

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("liger_kernel.ops.fused_linear_cross_entropy")


@pytest.mark.execution_timeout(180)
@pytest.mark.parametrize("mesh_shape", [(2,), (2, 2)], ids=["fsdp", "hsdp"])
def test_fsdp_loss_matches_unsharded_reference_cuda(mesh_shape, tmp_path):
    if (
        not dist.is_available()
        or not dist.is_nccl_available()
        or torch.cuda.device_count() < math.prod(mesh_shape)
    ):
        pytest.skip("requires NCCL and one CUDA device per rank")
    mp.spawn(
        _check_gradients_and_updates,
        args=(mesh_shape, "cuda", (tmp_path / "store").as_uri()),
        nprocs=math.prod(mesh_shape),
        join=True,
    )
