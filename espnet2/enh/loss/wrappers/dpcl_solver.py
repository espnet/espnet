from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class DPCLSolver(AbsLossWrapper):
    def __init__(self, criterion: AbsEnhLoss, weight=1.0):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(self, ref, inf, others={}):
        """A naive DPCL solver

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]
            others (List): other data included in this solver
                e.g. "tf_embedding" learned embedding of all T-F bins (B, T * F, D)

        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation
            stats: (dict), for collecting training status
            others: reserved
        """
        assert "tf_embedding" in others

        loss = self.criterion(ref, others["tf_embedding"]).mean()

        stats = dict()
        stats[self.criterion.name] = loss.detach()

        return loss.mean(), stats, {}
