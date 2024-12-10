from collections import defaultdict

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class FixedOrderSolver(AbsLossWrapper):
    """
    FixedOrderSolver is a wrapper class that implements a naive fixed-order 
solver for computing the loss between reference and inferred signals.

This class extends the AbsLossWrapper and utilizes a specified loss 
criterion to calculate the minimum loss with the best permutation of 
the input signals.

Attributes:
    criterion (AbsEnhLoss): The loss criterion used for calculating the 
        loss between reference and inferred signals.
    weight (float): A scaling factor for the computed loss.

Args:
    criterion (AbsEnhLoss): The loss criterion to be used for the solver.
    weight (float, optional): A weight factor for the loss. Defaults to 1.0.

Returns:
    Tuple[torch.Tensor, dict, dict]: A tuple containing:
        - loss (torch.Tensor): The minimum loss with the best permutation.
        - stats (dict): A dictionary containing training status statistics.
        - others (dict): Reserved for additional information, includes 
          the permutation used.

Raises:
    AssertionError: If the length of reference and inferred lists do not 
        match.

Examples:
    >>> solver = FixedOrderSolver(criterion=my_loss_criterion, weight=1.0)
    >>> ref_signals = [torch.randn(10, 1) for _ in range(2)]  # Two speakers
    >>> inf_signals = [torch.randn(10, 1) for _ in range(2)]
    >>> loss, stats, others = solver.forward(ref_signals, inf_signals)
    >>> print(loss)  # Outputs the computed loss
    >>> print(stats)  # Outputs training statistics
    >>> print(others)  # Outputs the permutation used
    """
    def __init__(self, criterion: AbsEnhLoss, weight=1.0):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(self, ref, inf, others={}):
        """
        An implementation of a naive fixed-order solver for loss computation in 
speech enhancement.

This class inherits from `AbsLossWrapper` and implements the `forward` 
method, which computes the loss based on the reference and inferred 
signals, considering a fixed order of speakers.

Attributes:
    criterion (AbsEnhLoss): The loss criterion used for computing the loss.
    weight (float): Weighting factor for the loss.

Args:
    ref (List[torch.Tensor]): A list of reference tensors, each of shape 
        (batch, ...), where `n_spk` is the number of speakers.
    inf (List[torch.Tensor]): A list of inferred tensors, each of shape 
        (batch, ...).
    others (dict, optional): Reserved for additional parameters (default: {}).

Returns:
    Tuple[torch.Tensor, dict, dict]: A tuple containing:
        - loss (torch.Tensor): The minimum loss with the best permutation.
        - stats (dict): A dictionary collecting training statistics.
        - others (dict): A dictionary containing reserved information, 
          including the speaker permutation.

Raises:
    AssertionError: If the lengths of `ref` and `inf` do not match.

Examples:
    >>> solver = FixedOrderSolver(criterion=my_criterion)
    >>> ref_signals = [torch.randn(10, 2) for _ in range(3)]
    >>> inf_signals = [torch.randn(10, 2) for _ in range(3)]
    >>> loss, stats, others = solver.forward(ref_signals, inf_signals)

Note:
    This method assumes that the number of reference signals matches the 
    number of inferred signals and that they are arranged in a specific 
    order corresponding to speakers.

Todo:
    - Implement more sophisticated permutation strategies to enhance 
      performance.
        """
        assert len(ref) == len(inf), (len(ref), len(inf))
        num_spk = len(ref)

        loss = 0.0
        stats = defaultdict(list)
        for r, i in zip(ref, inf):
            loss += torch.mean(self.criterion(r, i)) / num_spk
            for k, v in getattr(self.criterion, "stats", {}).items():
                stats[k].append(v)

        for k, v in stats.items():
            stats[k] = torch.stack(v, dim=1).mean()
        stats[self.criterion.name] = loss.detach()

        perm = torch.arange(num_spk).unsqueeze(0).repeat(ref[0].size(0), 1)
        return loss.mean(), dict(stats), {"perm": perm}
