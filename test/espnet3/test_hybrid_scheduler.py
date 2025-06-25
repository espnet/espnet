# import pytest
# import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, LinearLR

from espnet3.trainer import HybridLRS, HybridOptim

# ===============================================================
# Test Case Summary for HybridLRS
# ===============================================================
#
# Basic Wrapper Tests
# | Test Name                          | Description                                                              | # noqa: E501
# |-----------------------------------|--------------------------------------------------------------------------| # noqa: E501
# | test_hybrid_lrs_basic_attributes  | Validates optimizer, scheduler, and index assignment                    | # noqa: E501
# | test_getattr_forwarding           | Checks that method calls are correctly forwarded to lr_scheduler        | # noqa: E501
#


def create_optimizers_and_schedulers():
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 5)

    opt1 = SGD(model1.parameters(), lr=0.1)
    opt2 = Adam(model2.parameters(), lr=0.01)
    hybrid = HybridOptim([opt1, opt2])

    sched1 = StepLR(opt1, step_size=5, gamma=0.1)
    sched2 = LinearLR(opt2)

    return hybrid, [sched1, sched2]


def test_hybrid_lrs_basic_attributes():
    """Test Hybrid LRS.

    L001: Check that HybridLRS stores optimizer, idx, scheduler correctly
    """
    hybrid, scheds = create_optimizers_and_schedulers()
    schedulers = [
        HybridLRS(hybrid_optimizer=hybrid, lr_scheduler=sch, optimizer_idx=idx)
        for idx, sch in enumerate(scheds)
    ]

    for i, sch in enumerate(schedulers):
        assert sch.optimizer == hybrid
        assert sch.idx == i
        assert sch.lr_scheduler == scheds[i]
