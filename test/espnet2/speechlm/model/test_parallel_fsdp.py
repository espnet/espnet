# Copyright 2026 Carnegie Mellon University
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""Compare ParallelLLM._loss gradients and updates with an unsharded model."""

import copy
import math
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from packaging.version import Version
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


class _TorchLinearCE:
    """Differentiable CPU replacement for the CUDA-only Liger operation."""

    @staticmethod
    def apply(hidden, weight, target, bias, ce_weight, ignore_index, *args):
        logits = torch.nn.functional.linear(hidden, weight, bias)
        loss = torch.nn.functional.cross_entropy(
            logits, target, weight=ce_weight, ignore_index=ignore_index, reduction="sum"
        )
        return loss, loss.new_zeros(()), None


class _LossHarness(nn.Module):
    """Call the real _loss with small modules in place of a pretrained backbone."""

    def __init__(self, loss_impl):
        super().__init__()
        self.trunk = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 12, bias=False)
        self.stream_emb = nn.Embedding(4, 4)
        self.num_stream = 4
        self.multimodal_vocab_range = (8, 12)
        self.register_buffer("vocab_weight", torch.ones(12))
        self._loss_impl = loss_impl
        self._loss_stats = {}

    def forward(self, hidden, target, mask, scale):
        return self._loss_impl(
            self, (self.trunk(hidden), None), (target, mask), scale=scale
        )


def _batch(rank, step, micro, device):
    generator = torch.Generator().manual_seed(100 * step + 10 * micro + rank)
    hidden = torch.randn(2, 5, 4, generator=generator).to(device) + rank / 2
    target = torch.randint(1, 12, (2, 5, 4), generator=generator).to(device)
    target[:, :, 1:] = torch.randint(8, 12, (2, 5, 3), generator=generator).to(device)
    mask = torch.ones_like(target, dtype=torch.float32)
    mask[:, 0] = 0
    if rank % 2:
        mask[0, 1] = 0  # Unequal token counts across DP ranks.
    return hidden, target, mask


def _check_gradients_and_updates(rank, mesh_shape, device_type, rendezvous):
    torch.set_num_threads(1)
    world_size = math.prod(mesh_shape)
    device = torch.device(device_type, rank) if device_type == "cuda" else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        from espnet2.speechlm.model.speechlm.lm import loss as loss_module
        from espnet2.speechlm.model.speechlm.lm import parallel

        config = SimpleNamespace(architectures=["_FSDPLossTestModel"])
        with (
            mock.patch.object(
                parallel.AutoConfig, "from_pretrained", return_value=config
            ),
            mock.patch.object(
                parallel.transformers, "_FSDPLossTestModel", nn.Module, create=True
            ),
        ):
            loss_impl = parallel.build_parallel_hf_class("unused")._loss

        # CUDA uses the real Liger kernel; CPU substitutes only that operation.
        kernel = (
            _TorchLinearCE
            if device_type == "cpu"
            else loss_module.LigerFusedLinearCrossEntropyFunction
        )
        with mock.patch.object(
            loss_module, "LigerFusedLinearCrossEntropyFunction", kernel
        ):
            torch.manual_seed(6503)
            model = _LossHarness(loss_impl).to(device)
            reference = copy.deepcopy(model)
            mesh = init_device_mesh(
                device_type,
                mesh_shape,
                mesh_dim_names=(
                    ("replicate", "shard") if len(mesh_shape) == 2 else ("shard",)
                ),
            )
            for module in (model.trunk, model.lm_head, model.stream_emb):
                fully_shard(module, mesh=mesh)
            fully_shard(model, mesh=mesh)
            # Match SpeechLM's explicit prefetch of the directly accessed weights.
            model.trunk.set_modules_to_forward_prefetch(
                [model.lm_head, model.stream_emb]
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            ref_optimizer = torch.optim.SGD(
                reference.parameters(), lr=0.1, momentum=0.9
            )
            tolerance = 2e-4 if device_type == "cuda" else 1e-6

            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                ref_optimizer.zero_grad(set_to_none=True)
                model._loss_stats = {}
                reference._loss_stats = {}
                grad_accum = 2
                for micro in range(grad_accum):
                    batches = [
                        _batch(r, step, micro, device) for r in range(world_size)
                    ]
                    global_count = sum(batch[2][:, 1:, 0].sum() for batch in batches)
                    for batch in batches:
                        reference(
                            *batch, scale=1 / (global_count * grad_accum)
                        ).backward()
                    is_last = micro == grad_accum - 1
                    model.set_requires_all_reduce(is_last)
                    model.set_is_last_backward(is_last)
                    model(
                        *batches[rank], scale=world_size / (global_count * grad_accum)
                    ).backward()

                gradient_errors = []
                for name, parameter in model.named_parameters():
                    expected = dict(reference.named_parameters())[name]
                    actual_grad = parameter.grad.full_tensor()
                    try:
                        torch.testing.assert_close(
                            actual_grad,
                            expected.grad,
                            rtol=tolerance,
                            atol=tolerance,
                        )
                    except AssertionError as error:
                        gradient_errors.append(f"{name} gradient: {error}")
                assert not gradient_errors, "\n".join(gradient_errors)
                optimizer.step()
                ref_optimizer.step()
                for name, parameter in model.named_parameters():
                    expected = dict(reference.named_parameters())[name]
                    torch.testing.assert_close(
                        parameter.full_tensor(),
                        expected,
                        rtol=tolerance,
                        atol=tolerance,
                        msg=lambda message, name=name: f"{name} update: {message}",
                    )
    finally:
        dist.destroy_process_group()


@pytest.mark.execution_timeout(180)
@pytest.mark.parametrize("mesh_shape", [(2,), (2, 2)], ids=["fsdp", "hsdp"])
def test_fsdp_loss_matches_unsharded_reference(mesh_shape, tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires Gloo")
    if Version(torch.__version__.split("+")[0]) < Version("2.11"):
        pytest.skip("CPU FSDP regression uses the SpeechLM PyTorch 2.11 runtime")
    mp.spawn(
        _check_gradients_and_updates,
        args=(mesh_shape, "cpu", (tmp_path / "store").as_uri()),
        nprocs=math.prod(mesh_shape),
        join=True,
    )
