from distutils.version import LooseVersion

import pytest
import torch

from espnet2.schedulers.noam_lr import NoamLR
from espnet2.schedulers.warmup_lr import WarmupLR


@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion("1.1.0"),
    reason="Require pytorch>=1.1.0",
)
def test_WarumupLR():
    linear = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(linear.parameters(), 0.1)
    sch = WarmupLR(opt)
    lr = opt.param_groups[0]["lr"]

    opt.step()
    sch.step()
    lr2 = opt.param_groups[0]["lr"]
    assert lr != lr2


@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion("1.1.0"),
    reason="Require pytorch>=1.1.0",
)
def test_WarumupLR_is_compatible_with_NoamLR():
    lr = 10
    model_size = 320
    warmup_steps = 25000

    linear = torch.nn.Linear(2, 2)
    noam_opt = torch.optim.SGD(linear.parameters(), lr)
    noam = NoamLR(noam_opt, model_size=model_size, warmup_steps=warmup_steps)
    new_lr = noam.lr_for_WarmupLR(lr)

    linear = torch.nn.Linear(2, 2)
    warmup_opt = torch.optim.SGD(linear.parameters(), new_lr)
    warmup = WarmupLR(warmup_opt)

    for i in range(3 * warmup_steps):
        warmup_opt.step()
        warmup.step()

        noam_opt.step()
        noam.step()

        lr1 = noam_opt.param_groups[0]["lr"]
        lr2 = warmup_opt.param_groups[0]["lr"]

        assert lr1 == lr2
