import pytest
import torch
import torch.nn.functional as F

from espnet2.enh.loss.criterions.tf_domain import (
    FrequencyDomainCrossEntropy,
    FrequencyDomainL1,
)
from espnet2.enh.loss.wrappers.pit_solver import PITSolver


@pytest.mark.parametrize("num_spk", [1, 2, 3])
@pytest.mark.parametrize("flexible_numspk", [True, False])
def test_PITSolver_forward(num_spk, flexible_numspk):

    batch = 2
    inf = [torch.rand(batch, 10, 100) for spk in range(num_spk)]
    ref = [inf[num_spk - spk - 1] for spk in range(num_spk)]  # reverse inf as ref
    solver = PITSolver(
        FrequencyDomainL1(), independent_perm=True, flexible_numspk=flexible_numspk
    )

    loss, stats, others = solver(ref, inf)
    perm = others["perm"]
    correct_perm = list(range(num_spk))
    correct_perm.reverse()
    assert perm[0].equal(torch.tensor(correct_perm))

    # test for independent_perm is False

    solver = PITSolver(
        FrequencyDomainL1(), independent_perm=False, flexible_numspk=flexible_numspk
    )
    loss, stats, others = solver(ref, inf, {"perm": perm})


@pytest.mark.parametrize("num_spk", [1, 2, 3])
@pytest.mark.parametrize("flexible_numspk", [True, False])
def test_PITSolver_tf_ce_forward(num_spk, flexible_numspk):

    batch = 2
    ncls = 100
    ref = [torch.randint(0, ncls, (batch, 10)) for spk in range(num_spk)]
    bias = [F.one_hot(y) for y in ref]
    bias = [F.pad(y, (0, ncls - y.size(-1))) for y in bias]
    inf = [torch.rand(batch, 10, ncls) + bias[spk] for spk in range(num_spk)]
    solver = PITSolver(
        FrequencyDomainCrossEntropy(),
        independent_perm=True,
        flexible_numspk=flexible_numspk,
    )

    loss, stats, others = solver(ref, inf)
    perm = others["perm"]
    correct_perm = list(range(num_spk))
    assert perm[0].equal(torch.tensor(correct_perm)), (perm, correct_perm)

    # test for independent_perm is False

    solver = PITSolver(
        FrequencyDomainCrossEntropy(),
        independent_perm=False,
        flexible_numspk=flexible_numspk,
    )
    loss, stats, others = solver(ref, inf, {"perm": perm})
