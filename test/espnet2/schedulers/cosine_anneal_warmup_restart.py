import pytest
import torch

from espnet2.schedulers.cosine_anneal_warmup_restart import (
    CosineAnnealingWarmupRestarts,
)


@pytest.mark.parametrize("first_cycle_steps", [10, 20])
def test_CosineAnnealingWarmupRestarts(first_cycle_steps):
    linear = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(linear.parameters(), 0.1)
    sch = CosineAnnealingWarmupRestarts(
        opt,
        first_cycle_steps,
        cycle_mult=1.0,
        max_lr=0.1,
        min_lr=0.001,
        warmup_steps=0,
        gamma=1.0,
    )
    lr = opt.param_groups[0]["lr"]

    opt.step()
    sch.step()
    lr2 = opt.param_groups[0]["lr"]
    assert lr != lr2
