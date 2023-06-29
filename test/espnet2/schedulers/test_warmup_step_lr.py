import numpy as np
import torch

from espnet2.schedulers.warmup_step_lr import WarmupStepLR


def test_WarmupStepLR():
    linear = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(linear.parameters(), 0.1)
    sch = WarmupStepLR(opt)
    lr = opt.param_groups[0]["lr"]

    opt.step()
    sch.step()
    lr2 = opt.param_groups[0]["lr"]
    assert lr != lr2

    sch.step_num = sch.warmup_steps + 1
    opt.step()
    sch.step()
    lr3 = opt.param_groups[0]["lr"]
    assert lr2 != lr3
