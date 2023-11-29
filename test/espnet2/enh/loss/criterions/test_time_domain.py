import pytest
import torch
from packaging.version import parse as V

from espnet2.enh.loss.criterions.time_domain import (
    CISDRLoss,
    MultiResL1SpecLoss,
    SDRLoss,
    SISNRLoss,
    SNRLoss,
    TimeDomainL1,
    TimeDomainMSE,
)

is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")


@pytest.mark.parametrize(
    "criterion_class", [CISDRLoss, SISNRLoss, SNRLoss, SDRLoss, MultiResL1SpecLoss]
)
def test_time_domain_criterion_forward(criterion_class):
    criterion = criterion_class()

    batch = 2
    inf = torch.rand(batch, 2000)
    ref = torch.rand(batch, 2000)

    loss = criterion(ref, inf)
    assert loss.shape == (batch,), "Invlid loss shape with " + criterion.name


@pytest.mark.parametrize("criterion_class", [TimeDomainL1, TimeDomainMSE])
@pytest.mark.parametrize("input_ch", [1, 2])
def test_time_domain_l1_l2_forward(criterion_class, input_ch):
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


@pytest.mark.parametrize("window_sz", [[512], [256, 512]])
@pytest.mark.parametrize("time_domain_weight", [0.5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("normalize_variance", [True, False])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_multi_res_l1_spec_loss_forward_backward(
    window_sz, time_domain_weight, dtype, normalize_variance, reduction
):
    if dtype == torch.float16 and not is_torch_1_12_1_plus:
        pytest.skip("Skip tests for dtype=torch.float16 due to lack of torch.complex32")
    criterion = MultiResL1SpecLoss(
        window_sz=window_sz,
        time_domain_weight=time_domain_weight,
        normalize_variance=normalize_variance,
        reduction=reduction,
    )

    batch = 2
    inf = torch.rand(batch, 2000, dtype=dtype, requires_grad=True)
    ref = torch.rand(batch, 2000, dtype=dtype)

    loss = criterion(ref, inf)
    loss.sum().backward()
    assert loss.shape == (batch,), "Invlid loss shape with " + criterion.name
