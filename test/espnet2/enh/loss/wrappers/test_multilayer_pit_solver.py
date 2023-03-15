import pytest
import torch

from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainL1
from espnet2.enh.loss.wrappers.multilayer_pit_solver import MultiLayerPITSolver


@pytest.mark.parametrize("num_spk", [1, 2, 3])
def test_MultiLayerPITSolver_forward_multi_layer(num_spk):
    batch = 2
    num_layers = 2
    # infs is a List of List (num_layer x num_speaker Tensors)
    infs = [
        [torch.rand(batch, 10, 100) for spk in range(num_spk)]
        for _ in range(num_layers)
    ]
    ref = [infs[-1][num_spk - spk - 1] for spk in range(num_spk)]  # reverse inf as ref
    solver = MultiLayerPITSolver(FrequencyDomainL1(), independent_perm=True)

    loss, stats, others = solver(ref, infs)
    perm = others["perm"]
    correct_perm = list(range(num_spk))
    correct_perm.reverse()
    assert perm[0].equal(torch.tensor(correct_perm))

    # test for independent_perm is False

    solver = MultiLayerPITSolver(FrequencyDomainL1(), independent_perm=False)
    loss, stats, others = solver(ref, infs, {"perm": perm})


@pytest.mark.parametrize("num_spk", [1, 2, 3])
def test_MultiLayerPITSolver_forward_single_layer(num_spk):
    batch = 2
    # inf is a List of Tensors
    inf = [torch.rand(batch, 10, 100) for spk in range(num_spk)]
    ref = [inf[num_spk - spk - 1] for spk in range(num_spk)]  # reverse inf as ref
    solver = MultiLayerPITSolver(FrequencyDomainL1(), independent_perm=True)

    loss, stats, others = solver(ref, inf)
    perm = others["perm"]
    correct_perm = list(range(num_spk))
    correct_perm.reverse()
    assert perm[0].equal(torch.tensor(correct_perm))

    # test for independent_perm is False

    solver = MultiLayerPITSolver(FrequencyDomainL1(), independent_perm=False)
    loss, stats, others = solver(ref, inf, {"perm": perm})
