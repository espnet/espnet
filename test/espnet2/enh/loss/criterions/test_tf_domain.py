import pytest
import torch
from packaging.version import parse as V
from torch_complex import ComplexTensor

from espnet2.enh.loss.criterions.tf_domain import (FrequencyDomainAbsCoherence,
                                                   FrequencyDomainCrossEntropy,
                                                   FrequencyDomainDPCL,
                                                   FrequencyDomainL1,
                                                   FrequencyDomainMSE)

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


@pytest.mark.parametrize("criterion_class", [FrequencyDomainL1, FrequencyDomainMSE])
@pytest.mark.parametrize(
    "mask_type", ["IBM", "IRM", "IAM", "PSM", "NPSM", "PSM^2", "CIRM"]
)
@pytest.mark.parametrize("compute_on_mask", [True, False])
@pytest.mark.parametrize("input_ch", [1, 2])
def test_tf_domain_criterion_forward(
    criterion_class, mask_type, compute_on_mask, input_ch
):

    criterion = criterion_class(compute_on_mask=compute_on_mask, mask_type=mask_type)
    complex_wrapper = torch.complex if is_torch_1_9_plus else ComplexTensor

    batch = 2
    shape = (batch, 10, 200) if input_ch == 1 else (batch, 10, input_ch, 200)
    ref_spec = [complex_wrapper(torch.rand(*shape), torch.rand(*shape))]
    mix_spec = complex_wrapper(torch.rand(*shape), torch.rand(*shape))
    noise_spec = complex_wrapper(torch.rand(*shape), torch.rand(*shape))

    if compute_on_mask:
        inf = [torch.rand(*shape)]
        ref = criterion.create_mask_label(mix_spec, ref_spec, noise_spec=noise_spec)
        loss = criterion(ref[0], inf[0])
    else:
        inf_spec = [complex_wrapper(torch.rand(*shape), torch.rand(*shape))]
        loss = criterion(ref_spec[0], inf_spec[0])

    assert loss.shape == (batch,), "Invlid loss shape with " + criterion.name


@pytest.mark.parametrize("input_ch", [1, 2])
def test_tf_coh_criterion_forward(input_ch):

    criterion = FrequencyDomainAbsCoherence()
    complex_wrapper = torch.complex if is_torch_1_9_plus else ComplexTensor

    batch = 2
    shape = (batch, 10, 200) if input_ch == 1 else (batch, 10, input_ch, 200)
    inf_spec = complex_wrapper(torch.rand(*shape), torch.rand(*shape))
    ref_spec = complex_wrapper(torch.rand(*shape), torch.rand(*shape))

    loss = criterion(ref_spec, inf_spec)
    assert loss.shape == (batch,), "Invlid loss shape with " + criterion.name


@pytest.mark.parametrize("input_ch", [1, 2])
def test_tf_coh_criterion_invalid_forward(input_ch):

    criterion = FrequencyDomainAbsCoherence()
    complex_wrapper = torch.complex if is_torch_1_9_plus else ComplexTensor

    batch = 2
    shape = (batch, 10, 200) if input_ch == 1 else (batch, 10, input_ch, 200)
    inf_spec = complex_wrapper(torch.rand(*shape), torch.rand(*shape))
    ref_spec = complex_wrapper(torch.rand(*shape), torch.rand(*shape))

    with pytest.raises(ValueError):
        criterion(ref_spec.real, inf_spec)

    with pytest.raises(ValueError):
        if input_ch == 1:
            criterion(ref_spec[0], inf_spec[0])
        else:
            criterion(ref_spec[0, 0], inf_spec[0, 0])


@pytest.mark.parametrize("input_ch", [1, 2])
def test_tf_ce_criterion_forward(input_ch):

    criterion = FrequencyDomainCrossEntropy()

    batch = 2
    ncls = 200
    shape = (batch, 10, ncls) if input_ch == 1 else (batch, 10, input_ch, ncls)
    label_shape = (batch, 10) if input_ch == 1 else (batch, 10, input_ch)
    inf_spec = torch.rand(*shape)
    ref_spec = torch.randint(0, ncls, label_shape)

    loss = criterion(ref_spec, inf_spec)
    assert loss.shape == (batch,), "Invlid loss shape with " + criterion.name


@pytest.mark.parametrize("loss_type", ["dpcl", "mdc"])
def test_tf_dpcl_loss_criterion_forward(loss_type):

    criterion = FrequencyDomainDPCL(loss_type=loss_type)

    batch = 2
    inf = torch.rand(batch, 10 * 200, 40)
    ref_spec = [
        ComplexTensor(torch.rand(batch, 10, 200), torch.rand(batch, 10, 200)),
        ComplexTensor(torch.rand(batch, 10, 200), torch.rand(batch, 10, 200)),
        ComplexTensor(torch.rand(batch, 10, 200), torch.rand(batch, 10, 200)),
    ]

    ref = [abs(r) for r in ref_spec]

    loss = criterion(ref, inf)
    assert loss.shape == (batch,), "Invlid loss shape with " + criterion.name
