import itertools
from typing import Dict, List, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import einsum as complex_einsum
from espnet2.enh.layers.complex_utils import stack as complex_stack
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class MixITSolver(AbsLossWrapper):
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
        """MixIT solver.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...] x n_est
        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation
            stats: dict, for collecting training status
            others: dict, in this PIT solver, permutation order will be returned
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
