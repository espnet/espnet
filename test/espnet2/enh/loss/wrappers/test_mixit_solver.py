import pytest
import torch
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor

from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainL1
from espnet2.enh.loss.criterions.time_domain import TimeDomainL1
from espnet2.enh.loss.wrappers.mixit_solver import MixITSolver


@pytest.mark.parametrize("inf_num, time_domain", [(4, True), (4, False)])
def test_MixITSolver_forward(inf_num, time_domain):

    batch = 2
    if time_domain:
        solver = MixITSolver(TimeDomainL1())

        inf = [torch.rand(batch, 100) for _ in range(inf_num)]
        # 2 speaker's reference
        ref = [torch.zeros(batch, 100), torch.zeros(batch, 100)]
    else:
        solver = MixITSolver(FrequencyDomainL1())

        inf = [torch.rand(batch, 100, 10, 10) for _ in range(inf_num)]
        # 2 speaker's reference
        ref = [torch.zeros(batch, 100, 10, 10), torch.zeros(batch, 100, 10, 10)]

    ref[0][0] = inf[2][0] + inf[3][0]  # sample1, speaker 1
    ref[1][0] = inf[0][0] + inf[1][0]  # sample1, speaker 2
    ref[0][1] = inf[0][1] + inf[3][1]  # sample2, speaker 1
    ref[1][1] = inf[1][1] + inf[2][1]  # sample2, speaker 2

    loss, stats, others = solver(ref, inf)
    perm = others["perm"]
    correct_perm1 = (
        F.one_hot(
            torch.tensor([1, 1, 0, 0], dtype=torch.int64),
            num_classes=inf_num // 2,
        )
        .transpose(1, 0)
        .float()
    )
    assert perm[0].equal(torch.tensor(correct_perm1))

    correct_perm2 = (
        F.one_hot(
            torch.tensor([0, 1, 1, 0], dtype=torch.int64),
            num_classes=inf_num // 2,
        )
        .transpose(1, 0)
        .float()
    )
    assert perm[1].equal(torch.tensor(correct_perm2))


@pytest.mark.parametrize("inf_num", [4])
def test_MixITSolver_complex_tensor_forward(inf_num):

    batch = 2
    solver = MixITSolver(FrequencyDomainL1())

    inf = [
        ComplexTensor(
            torch.rand(batch, 100, 10, 10),
            torch.rand(batch, 100, 10, 10),
        )
        for _ in range(inf_num)
    ]
    # 2 speaker's reference
    ref = [
        ComplexTensor(
            torch.zeros(batch, 100, 10, 10),
            torch.zeros(batch, 100, 10, 10),
        )
        for _ in range(inf_num // 2)
    ]

    ref[0][0] = inf[2][0] + inf[3][0]  # sample1, speaker 1
    ref[1][0] = inf[0][0] + inf[1][0]  # sample1, speaker 2
    ref[0][1] = inf[0][1] + inf[3][1]  # sample2, speaker 1
    ref[1][1] = inf[1][1] + inf[2][1]  # sample2, speaker 2

    loss, stats, others = solver(ref, inf)
    perm = others["perm"]
    correct_perm1 = (
        F.one_hot(
            torch.tensor([1, 1, 0, 0], dtype=torch.int64),
            num_classes=inf_num // 2,
        )
        .transpose(1, 0)
        .float()
    )
    assert perm[0].equal(torch.tensor(correct_perm1))

    correct_perm2 = (
        F.one_hot(
            torch.tensor([0, 1, 1, 0], dtype=torch.int64),
            num_classes=inf_num // 2,
        )
        .transpose(1, 0)
        .float()
    )
    assert perm[1].equal(torch.tensor(correct_perm2))
