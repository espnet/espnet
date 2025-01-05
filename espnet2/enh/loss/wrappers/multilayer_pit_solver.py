from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.loss.wrappers.pit_solver import PITSolver


class MultiLayerPITSolver(AbsLossWrapper):
    """
        Multi-Layer Permutation Invariant Training Solver.

    This class computes the Permutation Invariant Training (PIT) loss using
    inferences from multiple layers and a single reference. It also supports
    single inference and single reference during the evaluation stage.

    Attributes:
        criterion (AbsEnhLoss): An instance of AbsEnhLoss used for computing loss.
        weight (float): Weight (between 0 and 1) of the current loss for
            multi-task learning.
        independent_perm (bool): If True, performs PIT in forward to find the
            best permutation; if False, inherits the permutation from the last
            LossWrapper output.
        layer_weights (Optional[List[float]]): Weights for each layer; if not
            None, the loss of each layer will be weighted-summed using the
            specified weights.
        solver (PITSolver): Instance of PITSolver to handle the PIT logic.

    Args:
        criterion (AbsEnhLoss): An instance of AbsEnhLoss.
        weight (float): Weight (between 0 and 1) of the current loss for
            multi-task learning. Defaults to 1.0.
        independent_perm (bool): If True, PIT will be performed in forward
            to find the best permutation; if False, inherits permutation from
            the last LossWrapper output. Defaults to True.
        layer_weights (Optional[List[float]]): Weights for each layer. If
            not None, the loss of each layer will be weighted-summed using
            the specified weights. Defaults to None.

    Methods:
        forward(ref, infs, others={}): Computes the minimum PIT loss with
            the best permutation and returns the loss, statistics, and
            permutation order.

    Examples:
        >>> criterion = SomeCriterion()
        >>> solver = MultiLayerPITSolver(criterion, weight=0.5)
        >>> ref = [torch.randn(10, 2), torch.randn(10, 2)]
        >>> infs = [[torch.randn(10, 2)], [torch.randn(10, 2)]]
        >>> loss, stats, others = solver.forward(ref, infs)

    Note:
        Be cautious about the ordering of loss wrappers defined in the YAML
        config when setting independent_perm to False.

    Todo:
        - Implement additional loss functions as needed for further
          flexibility.
    """

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
        """
            Permutation Invariant Training Solver.

        This method computes the minimum loss with the best permutation based on the
        provided references and inferences. It supports both single-layer and
        multi-layer cases, allowing for flexible handling of audio source separation
        tasks.

        Args:
            ref (List[torch.Tensor]): A list of tensors representing the reference
                signals, structured as [(batch, ...), ...] for n_spk.
            infs (Union[List[torch.Tensor], List[List[torch.Tensor]]]): A list of
                tensors representing the inferences. In the single-layer case, it
                should be structured as [(batch, ...), ...]. In the multi-layer
                case, it should be structured as List[List[torch.Tensor]].
            others (dict, optional): Additional arguments for training status
                collection or any other necessary data. Defaults to an empty
                dictionary.

        Returns:
            Tuple[torch.Tensor, dict, dict]: A tuple containing:
                - loss (torch.Tensor): The computed minimum loss with the best
                  permutation.
                - stats (dict): A dictionary for collecting training status.
                - others (dict): A dictionary that returns the permutation order
                  used in this PIT solver.

        Examples:
            >>> ref = [torch.randn(10, 2), torch.randn(10, 2)]
            >>> infs = [torch.randn(10, 2), torch.randn(10, 2)]
            >>> solver = MultiLayerPITSolver(criterion)
            >>> loss, stats, others = solver.forward(ref, infs)

        Note:
            This function is designed to work with the MultiLayerPITSolver class,
            which handles the complexity of permutation invariant training.

        Raises:
            ValueError: If the shapes of `ref` and `infs` do not match in the
            single-layer case.
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
