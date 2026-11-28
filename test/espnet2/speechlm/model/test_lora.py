# Copyright 2026 Carnegie Mellon University
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from packaging.version import Version
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

pytest.importorskip("liger_kernel.ops.fused_linear_cross_entropy")
# Spawned ranks import the real trainer independently of the pytest shims.
pytest.importorskip("torchtitan.distributed.parallel_dims")

from espnet2.speechlm.model.speechlm.lora import apply_lora  # noqa: E402
from espnet2.speechlm.trainer.titan_trainer import TitanTrainer  # noqa: E402


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.q_proj(x)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block()])
        self.audio = nn.ModuleDict({"q_proj": nn.Linear(4, 4)})

    def forward(self, x):
        return self.model.layers[0](self.audio["q_proj"](x))


def _adapt(model):
    apply_lora(model, target_modules=["q_proj"], rank=2, alpha=4)
    return model


def _trainer(model, directory):
    trainer = TitanTrainer.__new__(TitanTrainer)
    trainer.model = model
    trainer.trainer_args = {"lora": {}, "lr_scheduler": {"warmup_steps": 0}}
    trainer.global_rank = 0
    trainer.max_step = 10
    trainer.global_step = 0
    trainer.output_dir = directory
    trainer._build_optimizer_scheduler()
    return trainer


def _step(trainer, x):
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model(x).square().mean().backward()
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer.global_step += 1


def test_only_adapters_train():
    model = _Decoder()
    initial = copy.deepcopy(model)
    x = torch.randn(3, 4)
    _adapt(model)
    torch.testing.assert_close(model(x), initial(x), rtol=0, atol=0)
    trainable = {k for k, p in model.named_parameters() if p.requires_grad}
    prefix = "model.layers.0.q_proj"
    assert trainable == {f"{prefix}.lora_A", f"{prefix}.lora_B"}
    trainer = _trainer(model, None)
    for _ in range(2):
        _step(trainer, x)
    for key, param in model.named_parameters():
        if key in trainable:
            assert param.grad.abs().sum() > 0
        else:
            assert param.grad is None
            assert torch.equal(param, initial.state_dict()[key])
    assert not torch.equal(model.eval()(x), initial(x))


@pytest.mark.parametrize(
    "changes",
    [
        {"target_modules": []},
        {"target_modules": "q_proj"},
        {"target_modules": ["q_proj", "typo"]},
        {"rank": 0},
        {"alpha": float("nan")},
        {"dropout": 1},
    ],
)
def test_invalid_lora_config_leaves_base_intact(changes):
    model = _Decoder()
    with pytest.raises(ValueError, match="LoRA"):
        apply_lora(
            model, **({"target_modules": ["q_proj"], "rank": 2, "alpha": 4} | changes)
        )
    assert all(param.requires_grad for param in model.parameters())


def test_tied_projection_rejected():
    model = _Decoder()
    model.audio["q_proj"].weight = model.model.layers[0].q_proj.weight
    with pytest.raises(ValueError, match="shared weight"):
        _adapt(model)


@pytest.mark.parametrize("native", [True, False], ids=["native", "dcp"])
def test_fresh_lora_load_requires_all_base_or_all_adapted_keys(tmp_path, native):
    original = _Decoder()
    model = _adapt(_Decoder())
    trainer = _trainer(model, tmp_path)

    def save(name, state):
        path = tmp_path / name
        if native:
            torch.save({"module": state}, path)
        else:
            dcp.save({"model": state}, checkpoint_id=path, no_dist=True)
        return path

    state = original.state_dict()
    trainer._load_checkpoint(save("base", state))
    for key, weight in state.items():
        assert torch.equal(weight, model.state_dict()[key])
    assert not trainer.optimizer.state
    assert torch.count_nonzero(model.model.layers[0].q_proj.lora_B) == 0

    _step(trainer, torch.randn(3, 4))
    restored = _trainer(_adapt(_Decoder()), tmp_path)
    adapted = model.state_dict()
    restored._load_checkpoint(save("all_adapters", adapted))
    for key, weight in adapted.items():
        assert torch.equal(weight, restored.model.state_dict()[key])
    for name, deleted in (
        ("missing_base", "audio.q_proj.bias"),
        ("partial_adapter", "model.layers.0.q_proj.lora_A"),
    ):
        incomplete = {key: val for key, val in adapted.items() if key != deleted}
        with pytest.raises(ValueError, match="complete base model"):
            trainer._load_checkpoint(save(name, incomplete))


def _check_distributed_resume(rank, directory, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (2,))

        def build():
            torch.manual_seed(123)
            model = _adapt(_Decoder())
            mixed = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            )
            model.model.layers[0] = checkpoint_wrapper(model.model.layers[0])
            fully_shard(model.model.layers[0], mesh=mesh, mp_policy=mixed)
            fully_shard(model, mesh=mesh, mp_policy=mixed)
            return _trainer(model, directory)

        baseline = build()
        baseline._load_checkpoint(directory / "base.pt")
        full = StateDictOptions(full_state_dict=True)
        loaded = get_model_state_dict(baseline.model, options=full)
        for key, value in torch.load(directory / "base.pt", weights_only=True).items():
            assert torch.equal(loaded[key], value)

        x = torch.randn(3, 4) + rank
        _step(baseline, x)
        baseline._save_checkpoint(baseline.global_step)
        resumed = build()
        resumed._load_checkpoint(None)
        assert resumed.global_step == 1
        assert resumed.lr_scheduler.state_dict() == baseline.lr_scheduler.state_dict()
        for current in (baseline, resumed):
            _step(current, x)
        expected = get_model_state_dict(baseline.model, options=full)
        actual = get_model_state_dict(resumed.model, options=full)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert torch.count_nonzero(actual["model.layers.0.q_proj.lora_B"]) > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.execution_timeout(180)
def test_fsdp_initialization_and_exact_lora_resume(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires CPU Gloo")
    if Version(torch.__version__.split("+")[0]) < Version("2.11"):
        pytest.skip("CPU FSDP uses the SpeechLM PyTorch 2.11 runtime")
    torch.save(_Decoder().state_dict(), tmp_path / "base.pt")
    mp.spawn(
        _check_distributed_resume,
        args=(tmp_path, (tmp_path / "store").as_uri()),
        nprocs=2,
    )
