from collections import defaultdict
from itertools import permutations

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class PITSolver(AbsLossWrapper):
    """
    Permutation Invariant Training Solver.

    This class implements a solver for permutation invariant training (PIT),
    which is used to handle the permutation of speakers in the input. It 
    calculates the loss for all permutations and selects the one with the 
    minimum loss, making it suitable for multi-speaker scenarios.

    Attributes:
        criterion (AbsEnhLoss): An instance of AbsEnhLoss that defines the loss 
            computation.
        weight (float): Weight (between 0 and 1) of the current loss for 
            multi-task learning.
        independent_perm (bool): If True, PIT will be performed in forward to 
            find the best permutation; if False, the permutation from the last 
            LossWrapper output will be inherited.
        flexible_numspk (bool): If True, num_spk will be taken from inf to 
            handle flexible numbers of speakers.

    Args:
        criterion (AbsEnhLoss): An instance of AbsEnhLoss.
        weight (float): Weight (between 0 and 1) of the current loss for 
            multi-task learning.
        independent_perm (bool): 
            If True, PIT will be performed in forward to find the best 
            permutation; if False, the permutation from the last LossWrapper 
            output will be inherited.
            NOTE (wangyou): Be cautious about the ordering of loss wrappers 
            defined in the yaml config if this argument is False.
        flexible_numspk (bool): 
            If True, num_spk will be taken from inf to handle flexible numbers 
            of speakers, as ref may include dummy data in this case.

    Methods:
        forward(ref, inf, others={}):
            Computes the forward pass for the PIT solver, returning the minimum 
            loss and the corresponding permutation.

    Examples:
        # Example usage:
        pit_solver = PITSolver(criterion=my_criterion, weight=0.5)
        loss, stats, perm = pit_solver.forward(reference, inference)

    Raises:
        AssertionError: If flexible_numspk is False and the number of reference 
            tensors does not match the number of inference tensors.
    """
    def __init__(
        self,
        criterion: AbsEnhLoss,
        weight=1.0,
        independent_perm=True,
        flexible_numspk=False,
    ):
        """Permutation Invariant Training Solver.

        Args:
            criterion (AbsEnhLoss): an instance of AbsEnhLoss
            weight (float): weight (between 0 and 1) of current loss
                for multi-task learning.
            independent_perm (bool):
                If True, PIT will be performed in forward to find the best permutation;
                If False, the permutation from the last LossWrapper output will be
                inherited.
                NOTE (wangyou): You should be careful about the ordering of loss
                    wrappers defined in the yaml config, if this argument is False.
            flexible_numspk (bool):
                If True, num_spk will be taken from inf to handle flexible numbers of
                speakers. This is because ref may include dummy data in this case.
        """
        super().__init__()
        self.criterion = criterion
        self.weight = weight
        self.independent_perm = independent_perm
        self.flexible_numspk = flexible_numspk

    def forward(self, ref, inf, others={}):
        """
        PITSolver forward method.

This method computes the loss for the Permutation Invariant Training (PIT) 
using the provided reference and inferred tensors. It evaluates different 
permutations to find the one that minimizes the loss.

Args:
    ref (List[torch.Tensor]): A list of tensors representing the reference 
        signals, with each tensor shaped as (batch, ...). The length of the 
        list should correspond to the number of speakers (n_spk).
    inf (List[torch.Tensor]): A list of tensors representing the inferred 
        signals, shaped as (batch, ...). The length of this list may vary 
        based on the `flexible_numspk` attribute.
    others (dict, optional): Additional parameters, which may include:
        - "perm": A predefined permutation order to be used if available.

Returns:
    tuple: A tuple containing:
        - loss (torch.Tensor): The minimum loss computed with the best 
            permutation.
        - stats (dict): A dictionary containing the collected training 
            statistics.
        - others (dict): A dictionary containing the permutation order used 
            in this PIT solver.

Raises:
    AssertionError: If `flexible_numspk` is False and the lengths of `ref` 
        and `inf` do not match.

Examples:
    >>> ref = [torch.randn(2, 5) for _ in range(3)]  # 3 speakers
    >>> inf = [torch.randn(2, 5) for _ in range(3)]  # 3 speakers
    >>> solver = PITSolver(criterion=my_criterion)
    >>> loss, stats, perm = solver.forward(ref, inf)
    
Note:
    The `independent_perm` argument controls whether to compute permutations 
    independently or to use the last permutation from the LossWrapper. If 
    set to False, be cautious about the ordering of loss wrappers defined in 
    the YAML configuration.

Todo:
    - Add support for other types of loss functions.
        """
        perm = others["perm"] if "perm" in others else None

        if not self.flexible_numspk:
            assert len(ref) == len(inf), (len(ref), len(inf))
            num_spk = len(ref)
        else:
            num_spk = len(inf)

        stats = defaultdict(list)

        def pre_hook(func, *args, **kwargs):
            ret = func(*args, **kwargs)
            for k, v in getattr(self.criterion, "stats", {}).items():
                stats[k].append(v)
            return ret

        def pair_loss(permutation):
            return sum(
                [
                    pre_hook(self.criterion, ref[s], inf[t])
                    for s, t in enumerate(permutation)
                ]
            ) / len(permutation)

        if self.independent_perm or perm is None:
            # computate permuatation independently
            device = ref[0].device
            all_permutations = list(permutations(range(num_spk)))
            losses = torch.stack([pair_loss(p) for p in all_permutations], dim=1)
            loss, perm_ = torch.min(losses, dim=1)
            perm = torch.index_select(
                torch.tensor(all_permutations, device=device, dtype=torch.long),
                0,
                perm_,
            )
            # remove stats from unused permutations
            for k, v in stats.items():
                # (B, num_spk * len(all_permutations), ...)
                new_v = torch.stack(v, dim=1)
                B, L, *rest = new_v.shape
                assert L == num_spk * len(all_permutations), (L, num_spk)
                new_v = new_v.view(B, L // num_spk, num_spk, *rest).mean(2)
                if new_v.dim() > 2:
                    shapes = [1 for _ in rest]
                    perm0 = perm_.view(perm_.shape[0], 1, *shapes).expand(-1, -1, *rest)
                else:
                    perm0 = perm_.unsqueeze(1)
                stats[k] = new_v.gather(1, perm0.to(device=new_v.device)).unbind(1)
        else:
            loss = torch.stack(
                [
                    torch.stack(
                        [
                            pre_hook(
                                self.criterion,
                                ref[s][batch].unsqueeze(0),
                                inf[t][batch].unsqueeze(0),
                            )
                            for s, t in enumerate(p)
                        ]
                    ).mean()
                    for batch, p in enumerate(perm)
                ]
            )

        loss = loss.mean()

        for k, v in stats.items():
            stats[k] = torch.stack(v, dim=1).mean()
        stats[self.criterion.name] = loss.detach()

        return loss.mean(), dict(stats), {"perm": perm}
