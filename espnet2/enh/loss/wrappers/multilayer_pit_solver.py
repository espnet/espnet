from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.loss.wrappers.pit_solver import PITSolver


class MultiLayerPITSolver(AbsLossWrapper):
    def __init__(
        self,
        criterion: AbsEnhLoss,
        weight=1.0,
        independent_perm=True,
        layer_weights=None,
    ):
        """Multi-Layer Permutation Invariant Training Solver.

        Compute the PIT loss given inferences of multiple layers and a single reference.
        It also support single inference and single reference in evaluation stage.

        Args:
            criterion (AbsEnhLoss): an instance of AbsEnhLoss
            weight (float): weight (between 0 and 1) of current loss
                for multi-task learning.
            independent_perm (bool):
                If True, PIT will be performed in forward to find the best permutation;
                If False, the permutation from the last LossWrapper output will be
                inherited.
                Note: You should be careful about the ordering of loss
                wrappers defined in the yaml config, if this argument is False.
            layer_weights (Optional[List[float]]): weights for each layer
                If not None, the loss of each layer will be weighted-summed using the
                specified weights.
        """
        super().__init__()
        self.criterion = criterion
        self.weight = weight
        self.independent_perm = independent_perm
        self.solver = PITSolver(criterion, weight, independent_perm)
        self.layer_weights = layer_weights

    def forward(self, ref, infs, others={}):
        """Permutation invariant training solver.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            infs (Union[List[torch.Tensor], List[List[torch.Tensor]]]):
                [(batch, ...), ...]

        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation
            stats: dict, for collecting training status
            others: dict, in this PIT solver, permutation order will be returned
        """
        losses = 0.0
        # In single-layer case, the model only estimates waveforms in the last layer.
        # The shape of infs is List[torch.Tensor]
        if not isinstance(infs[0], (tuple, list)) and len(infs) == len(ref):
            loss, stats, others = self.solver(ref, infs, others)
            losses = loss
        # In multi-layer case, weighted-sum the PIT loss of each layer
        # The shape of ins is List[List[torch.Tensor]]
        else:
            for idx, inf in enumerate(infs):
                loss, stats, others = self.solver(ref, inf, others)
                if self.layer_weights is not None:
                    losses = losses + loss * self.layer_weights[idx]
                else:
                    losses = losses + loss * (idx + 1) * (1.0 / len(infs))
            losses = losses / len(infs)
        return losses, stats, others
