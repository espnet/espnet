import torch

from espnet2.schedulers.piecewise_linear_warmup_lr import PiecewiseLinearWarmupLR


def test_PiecewiseLinearWarumupLR():
    linear = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(linear.parameters(), 0.1)
    sch = PiecewiseLinearWarmupLR(opt)
    lr = opt.param_groups[0]["lr"]

    opt.step()
    sch.step()
    lr2 = opt.param_groups[0]["lr"]
    assert lr != lr2
