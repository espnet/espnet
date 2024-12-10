from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class DPCLSolver(AbsLossWrapper):
    """
    DPCLSolver is a wrapper for a Deep Permutation Invariant Contrastive Learning 
(DPCL) loss function. This class inherits from the AbsLossWrapper and is 
designed to compute the minimum loss with the best permutation of the input 
data. It utilizes a given enhancement loss criterion for calculating the 
loss value.

Attributes:
    criterion (AbsEnhLoss): The loss criterion used for computing the loss.
    weight (float): A scaling factor for the loss.

Args:
    criterion (AbsEnhLoss): The enhancement loss criterion to be used.
    weight (float, optional): The weight for the loss. Defaults to 1.0.

Methods:
    forward(ref, inf, others={}):
        Computes the loss based on the reference and input tensors.

Returns:
    Tuple[torch.Tensor, dict, dict]: 
        - loss (torch.Tensor): The minimum loss with the best permutation.
        - stats (dict): A dictionary containing training status statistics.
        - others (dict): Reserved for future use.

Raises:
    AssertionError: If "tf_embedding" is not included in the others argument.

Examples:
    >>> criterion = SomeEnhLoss()
    >>> dpcl_solver = DPCLSolver(criterion)
    >>> ref = [torch.randn(2, 5), torch.randn(2, 5)]  # Two speakers
    >>> inf = [torch.randn(2, 5)]  # Single input
    >>> others = {"tf_embedding": torch.randn(2, 10, 5)}  # Example embedding
    >>> loss, stats, _ = dpcl_solver.forward(ref, inf, others)
    
Note:
    The "tf_embedding" should be included in the `others` argument for 
    proper functioning of the forward method.
    """
    def __init__(self, criterion: AbsEnhLoss, weight=1.0):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(self, ref, inf, others={}):
        """
        A naive DPCL solver for calculating the minimum loss with the best permutation.

This method computes the loss between the reference and inferred signals, utilizing
a learned embedding of time-frequency (T-F) bins. The DPCL solver aims to optimize
the loss based on permutations of the input signals.

Attributes:
    criterion (AbsEnhLoss): The criterion used to calculate the loss.
    weight (float): Weighting factor for the loss, default is 1.0.

Args:
    ref (List[torch.Tensor]): A list of tensors representing the reference signals,
        structured as [(batch, ...), ...] for n_spk speakers.
    inf (List[torch.Tensor]): A list of tensors representing the inferred signals,
        structured as [(batch, ...), ...].
    others (dict): Additional data required by this solver, must include:
        - "tf_embedding": A learned embedding of all T-F bins with shape (B, T * F, D).

Returns:
    Tuple[torch.Tensor, dict, dict]: A tuple containing:
        - loss (torch.Tensor): The minimum loss calculated with the best permutation.
        - stats (dict): A dictionary for collecting training status, includes:
            - criterion name and corresponding loss value.
        - others (dict): Reserved for future use.

Raises:
    AssertionError: If "tf_embedding" is not present in the others argument.

Examples:
    >>> solver = DPCLSolver(criterion=my_criterion)
    >>> loss, stats, _ = solver.forward(reference_signals, inferred_signals, 
    ...                                  others={"tf_embedding": tf_embeddings})

Note:
    Ensure that the "tf_embedding" is correctly provided in the others dictionary 
    for the forward method to function properly.
        """
        assert "tf_embedding" in others

        loss = self.criterion(ref, others["tf_embedding"]).mean()

        stats = dict()
        stats[self.criterion.name] = loss.detach()

        return loss.mean(), stats, {}
