import torch

from espnet2.schedulers.noam_lr import NoamLR


def test_NoamLR():
    l = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(l.parameters(), 0.1)
    sch = NoamLR(opt)
    opt.step()
    sch.step()
