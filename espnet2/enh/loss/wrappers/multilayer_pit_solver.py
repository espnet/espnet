from itertools import permutations

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.loss.wrappers.pit_solver import PITSolver


class MultiLayerPITSolver(AbsLossWrapper):
    def __init__(self, criterion: AbsEnhLoss, weight=1.0, independent_perm=True,):
        super().__init__()
        self.criterion = criterion
        self.weight = weight
        self.independent_perm = independent_perm
        self.solver = PITSolver(criterion, weight, independent_perm)

    def forward(self, ref, infs, others={}):
        """Permutation invariant training solver.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]

        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation
            stats: dict, for collecting training status
            others: dict, in this PIT solver, permutation order will be returned
        """
        losses = 0.0
        for idx, inf in enumerate(infs):
            loss, stats, others = self.solver(ref, inf, others)
            losses = loss * (idx + 1) * (1.0 / len(infs))
        losses = losses / len(infs)
        return losses, stats, others
        
