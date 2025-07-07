import torch

from espnet2.schedulers.exponential_decay_warmup import (
    ExponentialDecayWarmup,
)


def test_ExponentialDecayWarmup():
    linear = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(linear.parameters(), 0.1)
    sch = ExponentialDecayWarmup(
        opt,
        max_lr=0.1,
        min_lr=0.001,
        total_steps=100,
        warmup_steps=10,
        warm_from_zero=True,
    )
    lr = opt.param_groups[0]["lr"]

    opt.step()
    sch.step()
    lr2 = opt.param_groups[0]["lr"]
    assert lr != lr2
