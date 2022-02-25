import pytest
import torch

from torch_complex import ComplexTensor

from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainL1
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainMSE
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainDPCL


@pytest.mark.parametrize("criterion_class", [FrequencyDomainDPCL])
@pytest.mark.parametrize("mask_type", ["IBM"])
@pytest.mark.parametrize("compute_on_mask", [False])
@pytest.mark.parametrize("loss_type", ["dpcl", "mdc"])
def test_tf_domain_criterion_forward(criterion_class, mask_type, compute_on_mask, loss_type):

    criterion = criterion_class(compute_on_mask=compute_on_mask, mask_type=mask_type, loss_type=loss_type)

    batch = 2
    inf = torch.rand(batch, 10*200, 40)
    ref_spec = [ComplexTensor(torch.rand(batch, 10, 200), torch.rand(batch, 10, 200)), ComplexTensor(torch.rand(batch, 10, 200), torch.rand(batch, 10, 200)), ComplexTensor(torch.rand(batch, 10, 200), torch.rand(batch, 10, 200))]

    ref = [abs(r) for r in ref_spec]

    loss = criterion(ref, inf)
    assert loss.shape == (batch,)
