import pytest
import torch

from espnet2.enh.loss.criterions.time_domain import (
    CISDRLoss,
    MultiResL1SpecLoss,
    SDRLoss,
    SISNRLoss,
    SNRLoss,
    TimeDomainL1,
    TimeDomainMSE,
)


@pytest.mark.parametrize(
    "criterion_class", [CISDRLoss, SISNRLoss, SNRLoss, SDRLoss, MultiResL1SpecLoss]
)
def test_tf_domain_criterion_forward(criterion_class):

    criterion = criterion_class()

    batch = 2
    inf = torch.rand(batch, 2000)
    ref = torch.rand(batch, 2000)

    loss = criterion(ref, inf)
    assert loss.shape == (batch,), "Invlid loss shape with " + criterion.name


@pytest.mark.parametrize("criterion_class", [TimeDomainL1, TimeDomainMSE])
@pytest.mark.parametrize("input_ch", [1, 2])
def test_tf_domain_l1_l2_forward(criterion_class, input_ch):

    criterion = criterion_class()

    batch = 2
    shape = (batch, 200) if input_ch == 1 else (batch, 200, input_ch)
    inf = torch.rand(*shape)
    ref = torch.rand(*shape)

    loss = criterion(ref, inf)
    assert loss.shape == (batch,), "Invlid loss shape with " + criterion.name

    with pytest.raises(ValueError):
        if input_ch == 1:
            loss = criterion(ref[..., None, None], inf[..., None, None])
        else:
            loss = criterion(ref[..., None], inf[..., None])
