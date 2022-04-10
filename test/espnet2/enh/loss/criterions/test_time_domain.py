import pytest
import torch

from espnet2.enh.loss.criterions.time_domain import CISDRLoss
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.criterions.time_domain import SNRLoss
from espnet2.enh.loss.criterions.time_domain import TimeDomainL1
from espnet2.enh.loss.criterions.time_domain import TimeDomainMSE


@pytest.mark.parametrize(
    "criterion_class", [CISDRLoss, SISNRLoss, SNRLoss, TimeDomainL1, TimeDomainMSE]
)
def test_tf_domain_criterion_forward(criterion_class):

    criterion = criterion_class()

    batch = 2
    inf = torch.rand(batch, 2000)
    ref = torch.rand(batch, 2000)

    loss = criterion(ref, inf)
    assert loss.shape == (batch,)
