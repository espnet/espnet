import itertools
from typing import Dict, List, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import einsum as complex_einsum
from espnet2.enh.layers.complex_utils import stack as complex_stack
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class MixITSolver(AbsLossWrapper):
    """
    MixITSolver is a Mixture Invariant Training Solver that extends the 
AbsLossWrapper class for multi-task learning in speech enhancement.

This solver computes the loss by exploring all possible permutations of 
the estimated sources against the reference sources to find the minimum 
loss. It is designed to work with both real and complex tensors.

Attributes:
    criterion (AbsEnhLoss): An instance of AbsEnhLoss used to compute the loss.
    weight (float): Weight (between 0 and 1) of the current loss for 
        multi-task learning.

Args:
    criterion (AbsEnhLoss): An instance of AbsEnhLoss.
    weight (float): Weight (between 0 and 1) of the current loss for 
        multi-task learning.

Returns:
    Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        - loss (torch.Tensor): Minimum loss with the best permutation.
        - stats (Dict[str, torch.Tensor]): A dictionary for collecting 
          training status.
        - others (Dict[str, torch.Tensor]): In this PIT solver, the 
          permutation order will be returned.

Examples:
    >>> criterion = SomeEnhLoss()  # Replace with an actual criterion
    >>> mixit_solver = MixITSolver(criterion, weight=0.5)
    >>> ref = [torch.randn(2, 10), torch.randn(2, 10)]
    >>> inf = [torch.randn(2, 10), torch.randn(2, 10), 
                torch.randn(2, 10), torch.randn(2, 10)]
    >>> loss, stats, others = mixit_solver(ref, inf)

Note:
    This class requires that the input tensors are either all real 
    or all complex. The permutation of the estimated sources is 
    determined to minimize the loss against the reference sources.

Todo:
    - Add support for additional loss criteria.
    """
    def __init__(
        self,
        criterion: AbsEnhLoss,
        weight: float = 1.0,
    ):
        """Mixture Invariant Training Solver.

        Args:
            criterion (AbsEnhLoss): an instance of AbsEnhLoss
            weight (float): weight (between 0 and 1) of current loss
                for multi-task learning.
        """
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    @property
    def name(self):
        return "mixit"

    def _complex_einsum(self, equation, *operands):
        for op in operands:
            if not isinstance(op, ComplexTensor):
                op = ComplexTensor(op, torch.zeros_like(op))
        return complex_einsum(equation, *operands)

    def forward(
        self,
        ref: Union[List[torch.Tensor], List[ComplexTensor]],
        inf: Union[List[torch.Tensor], List[ComplexTensor]],
        others: Dict = {},
    ):
        """
        MixIT solver for calculating the minimum loss with the best permutation.

        This method computes the loss for a mixture of inputs using the MixIT 
        approach. It evaluates all possible permutations of the reference and 
        estimated tensors to find the one that yields the minimum loss. The 
        function also returns statistics for training status and the permutation 
        order used in the loss calculation.

        Args:
            ref (Union[List[torch.Tensor], List[ComplexTensor]]): 
                A list of reference tensors, where each tensor has shape 
                [(batch, ...), ...] for n_spk speakers.
            inf (Union[List[torch.Tensor], List[ComplexTensor]]): 
                A list of estimated tensors, where each tensor has shape 
                [(batch, ...), ...] for n_est estimates.
            others (Dict, optional): 
                A dictionary for any additional parameters. Defaults to an empty 
                dictionary.

        Returns:
            Tuple[torch.Tensor, Dict, Dict]:
                - loss (torch.Tensor): The minimum loss calculated with the 
                  best permutation.
                - stats (Dict): A dictionary containing training status metrics.
                - others (Dict): A dictionary containing the permutation order 
                  used in this PIT solver.

        Examples:
            >>> ref = [torch.rand(2, 3), torch.rand(2, 3)]  # Two reference tensors
            >>> inf = [torch.rand(2, 3), torch.rand(2, 3), 
            ...          torch.rand(2, 3), torch.rand(2, 3)]  # Four estimates
            >>> loss, stats, others = mixit_solver.forward(ref, inf)
            >>> print(loss)  # Outputs the minimum loss
            >>> print(others['perm'])  # Outputs the permutation order

        Note:
            Ensure that the input tensors are either all complex or all real 
            tensors for proper computation. The function asserts that the types 
            of `ref` and `inf` match.

        Raises:
            AssertionError: If the number of estimated tensors is not double 
            the number of reference tensors or if the input types do not match.

        Todo:
            - Add support for more complex loss calculations.
            - Implement caching for previously computed permutations to 
            improve efficiency.
        """
        num_inf = len(inf)
        num_ref = num_inf // 2
        device = ref[0].device

        is_complex = isinstance(ref[0], ComplexTensor)
        assert is_complex == isinstance(inf[0], ComplexTensor)

        if not is_complex:
            ref_tensor = torch.stack(ref[:num_ref], dim=1)  # (batch, num_ref, ...)
            inf_tensor = torch.stack(inf, dim=1)  # (batch, num_inf, ...)

            einsum_fn = torch.einsum
        else:
            ref_tensor = complex_stack(ref[:num_ref], dim=1)  # (batch, num_ref, ...)
            inf_tensor = complex_stack(inf, dim=1)  # (batch, num_inf, ...)

            einsum_fn = self._complex_einsum

        # all permutation assignments:
        #   [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), ..., (1, 1, 1, 1)]
        all_assignments = list(itertools.product(range(num_ref), repeat=num_inf))
        all_mixture_matrix = torch.stack(
            [
                torch.nn.functional.one_hot(
                    torch.tensor(asm, dtype=torch.int64, device=device),
                    num_classes=num_ref,
                ).transpose(1, 0)
                for asm in all_assignments
            ],
            dim=0,
        ).to(
            inf_tensor.dtype
        )  # (num_ref ^ num_inf, num_ref, num_inf)

        # (num_ref ^ num_inf, batch, num_ref, seq_len, ...)
        if inf_tensor.dim() == 3:
            est_sum_mixture = einsum_fn("ari,bil->abrl", all_mixture_matrix, inf_tensor)
        elif inf_tensor.dim() > 3:
            est_sum_mixture = einsum_fn(
                "ari,bil...->abrl...", all_mixture_matrix, inf_tensor
            )

        losses = []
        for i in range(all_mixture_matrix.shape[0]):
            losses.append(
                sum(
                    [
                        self.criterion(ref_tensor[:, s], est_sum_mixture[i, :, s])
                        for s in range(num_ref)
                    ]
                )
                / num_ref
            )
        losses = torch.stack(losses, dim=0)  # (num_ref ^ num_inf, batch)

        loss, perm = torch.min(losses, dim=0)  # (batch)
        loss = loss.mean()
        perm = torch.index_select(all_mixture_matrix, 0, perm)

        if perm.is_complex():
            perm = perm.real

        stats = dict()
        stats[f"{self.criterion.name}_{self.name}"] = loss.detach()

        return loss.mean(), stats, {"perm": perm}
