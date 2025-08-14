# import pytest
# import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LinearLR, StepLR

from espnet3.trainer.multiple_optim import MultipleOptim
from espnet3.trainer.multiple_scheduler import MultipleScheduler

# ===============================================================
# Test Case Summary for MultipleScheduler
# ===============================================================
#
# Basic Wrapper Tests
# | Test Name                          | Description                                                              | # noqa: E501
# |-----------------------------------|--------------------------------------------------------------------------| # noqa: E501
# | test_multiple_lrs_basic_attributes  | Validates optimizer, scheduler, and index assignment                    | # noqa: E501
# | test_getattr_forwarding           | Checks that method calls are correctly forwarded to lr_scheduler        | # noqa: E501
#


def create_optimizers_and_schedulers():
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 5)

    opt1 = SGD(model1.parameters(), lr=0.1)
    opt2 = Adam(model2.parameters(), lr=0.01)
    mopts = MultipleOptim([opt1, opt2])

    sched1 = StepLR(opt1, step_size=5, gamma=0.1)
    sched2 = LinearLR(opt2)

    return mopts, [sched1, sched2]


def test_multiple_lrs_basic_attributes():
    """Test Multiple LRS.

    L001: Check that MultipleScheduler stores optimizer, idx, scheduler correctly
    """
    mopts, scheds = create_optimizers_and_schedulers()
    schedulers = [
        MultipleScheduler(multiple_optimizer=mopts, lr_scheduler=sch, optimizer_idx=idx)
        for idx, sch in enumerate(scheds)
    ]

    for i, sch in enumerate(schedulers):
        assert sch.optimizer == mopts
        assert sch.idx == i
        assert sch.lr_scheduler == scheds[i]
