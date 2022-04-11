import itertools

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class MixITSolver(AbsLossWrapper):
    def __init__(self, criterion: AbsEnhLoss, weight=1.0):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    @property
    def type(self):
        return "mixit"

    def forward(self, ref, inf, others={}):
        """MixIT solver.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...] x n_est

        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation
            stats: dict, for collecting training status
            others: dict, in this PIT solver, permutation order will be returned
        """
        num_ref, num_inf = len(ref), len(inf)
        device = ref[0].device

        ref_tensor = torch.stack(ref, dim=1)  # (batch, num_ref, ...)
        inf_tensor = torch.stack(inf, dim=1)  # (batch, num_inf, ...)

        # all permutation assignments: [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), ..., (1, 1, 1, 1)]
        all_assignments = list(itertools.product(range(num_ref), repeat=num_inf))
        all_mixture_matrix = torch.stack(
            [
                torch.nn.functional.one_hot(
                    torch.tensor(asm, dtype=torch.int64, device=device),
                    num_classes=num_ref,
                ).transpose(1, 0) for asm in all_assignments
            ],
            dim = 0,
        ).float()  # (num_ref ^ num_inf, num_ref, num_inf)

        def pair_loss(matrix):
            mix_estimated = torch.matmul(matrix[None], inf_tensor)
            return sum(
                [self.criterion(ref_tensor[:, i], mix_estimated[:, i]) for i in range(num_ref)]
            ) / num_ref

        losses = torch.stack(
            [pair_loss(matrix) for matrix in all_mixture_matrix],
            dim=1,
        )  # (batch, num_ref ^ num_inf)
        loss, perm = torch.min(losses, dim=1)
        perm = torch.index_select(all_mixture_matrix, 0, perm)

        loss = loss.mean()

        stats = dict()
        stats[f"{self.criterion.name}_{self.type}"] = loss.detach()

        return loss.mean(), stats, {"perm": perm}
