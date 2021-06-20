import torch

from espnet2.schedulers.noam_lr import NoamLR


def test_NoamLR():
    linear = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(linear.parameters(), 0.1)
    sch = NoamLR(opt)
    lr = opt.param_groups[0]["lr"]

    opt.step()
    sch.step()
    lr2 = opt.param_groups[0]["lr"]
    assert lr != lr2
