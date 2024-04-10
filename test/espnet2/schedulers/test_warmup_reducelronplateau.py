import torch

from espnet2.schedulers.warmup_reducelronplateau import WarmupReduceLROnPlateau


def test_WarmupReduceLROnPlateau():
    linear = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(linear.parameters(), 0.1)
    sch = WarmupReduceLROnPlateau(opt, mode="min", factor=0.1, patience=1, cooldown=0)
    lr = opt.param_groups[0]["lr"]

    opt.step()
    sch.step()
    lr2 = opt.param_groups[0]["lr"]
    assert lr != lr2

    sch.step_num = sch.warmup_steps + 1
    opt.step()
    sch.step(2.5)
    lr3 = opt.param_groups[0]["lr"]
    assert lr3 == lr2

    opt.step()
    sch.step(3.5)
    lr4 = opt.param_groups[0]["lr"]
    assert lr4 == lr3

    opt.step()
    sch.step(4.5)
    lr5 = opt.param_groups[0]["lr"]
    assert lr5 == lr4 * 0.1
