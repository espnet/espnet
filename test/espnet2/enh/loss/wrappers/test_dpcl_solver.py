import pytest
import torch

from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainDPCL
from espnet2.enh.loss.wrappers.dpcl_solver import DPCLSolver


@pytest.mark.parametrize("num_spk", [1, 2, 3])
def test_DPCLSolver_forward(num_spk):

    batch = 2
    o = {"tf_embedding": torch.rand(batch, 10 * 200, 40)}
    inf = [torch.rand(batch, 10, 200) for spk in range(num_spk)]
    ref = [inf[num_spk - spk - 1] for spk in range(num_spk)]  # reverse inf as ref
    solver = DPCLSolver(FrequencyDomainDPCL())

    loss, stats, others = solver(ref, inf, o)
