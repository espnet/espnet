import pytest
import torch

from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainL1
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver


@pytest.mark.parametrize("num_spk", [1, 2, 3])
def test_PITSolver_forward(num_spk):
    batch = 2
    inf = [torch.rand(batch, 10, 100) for spk in range(num_spk)]
    ref = [inf[num_spk - spk - 1] for spk in range(num_spk)]  # reverse inf as ref
    solver = FixedOrderSolver(FrequencyDomainL1())

    loss, stats, others = solver(ref, inf)
