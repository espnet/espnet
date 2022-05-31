import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class FixedOrderSolver(AbsLossWrapper):
    def __init__(self, criterion: AbsEnhLoss, weight=1.0):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(self, ref, inf, others={}):
        """An naive fixed-order solver

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]

        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation
            stats: dict, for collecting training status
            others: reserved
        """
        assert len(ref) == len(inf), (len(ref), len(inf))
        num_spk = len(ref)

        loss = 0.0

        for r, i in zip(ref, inf):
            loss += torch.mean(self.criterion(r, i)) / num_spk

        stats = dict()
        stats[self.criterion.name] = loss.detach()

        return loss.mean(), stats, {}
