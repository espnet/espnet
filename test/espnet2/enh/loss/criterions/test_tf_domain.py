import pytest
import torch

from torch_complex import ComplexTensor

from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainL1
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainMSE


@pytest.mark.parametrize("criterion_class", [FrequencyDomainL1, FrequencyDomainMSE])
@pytest.mark.parametrize(
    "mask_type", ["IBM", "IRM", "IAM", "PSM", "NPSM", "PSM^2", "CIRM"]
)
@pytest.mark.parametrize("compute_on_mask", [True, False])
def test_tf_domain_criterion_forward(criterion_class, mask_type, compute_on_mask):

    criterion = criterion_class(compute_on_mask=compute_on_mask, mask_type=mask_type)

    batch = 2
    inf = [torch.rand(batch, 10, 200)]
    ref_spec = [ComplexTensor(torch.rand(batch, 10, 200), torch.rand(batch, 10, 200))]
    mix_spec = ComplexTensor(torch.rand(batch, 10, 200), torch.rand(batch, 10, 200))

    if compute_on_mask:
        ref = criterion.create_mask_label(mix_spec, ref_spec)
    else:
        ref = [abs(r) for r in ref_spec]

    loss = criterion(ref[0], inf[0])
    assert loss.shape == (batch,)
