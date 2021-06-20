import numpy as np
import torch

from espnet2.schedulers.noam_lr import NoamLR
from espnet2.schedulers.warmup_lr import WarmupLR


def test_WarumupLR():
    linear = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(linear.parameters(), 0.1)
    sch = WarmupLR(opt)
    lr = opt.param_groups[0]["lr"]

    opt.step()
    sch.step()
    lr2 = opt.param_groups[0]["lr"]
    assert lr != lr2


def test_WarumupLR_is_compatible_with_NoamLR():
    lr = 10
    model_size = 32
    warmup_steps = 250

    linear = torch.nn.Linear(2, 2)
    noam_opt = torch.optim.SGD(linear.parameters(), lr)
    noam = NoamLR(noam_opt, model_size=model_size, warmup_steps=warmup_steps)
    new_lr = noam.lr_for_WarmupLR(lr)

    linear = torch.nn.Linear(2, 2)
    warmup_opt = torch.optim.SGD(linear.parameters(), new_lr)
    warmup = WarmupLR(warmup_opt, warmup_steps=warmup_steps)

    for i in range(3 * warmup_steps):
        warmup_opt.step()
        warmup.step()

        noam_opt.step()
        noam.step()

        lr1 = noam_opt.param_groups[0]["lr"]
        lr2 = warmup_opt.param_groups[0]["lr"]

        np.testing.assert_almost_equal(lr1, lr2)
