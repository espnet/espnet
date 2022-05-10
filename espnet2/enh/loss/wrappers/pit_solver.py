from collections import defaultdict
from itertools import permutations

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class PITSolver(AbsLossWrapper):
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
        """PITSolver forward.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]

        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation
            stats: dict, for collecting training status
            others: dict, in this PIT solver, permutation order will be returned
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
                # (B, len(all_permutations), ...)
                new_v = torch.stack(v, dim=1)
                if new_v.dim() > 2:
                    shapes = [1 for _ in range(new_v.dim() - 2)]
                    perm0 = perm_.view(perm_.shape[0], 1, *shapes).expand(
                        -1, -1, *new_v.shape[2:]
                    )
                else:
                    perm0 = perm_.unsqueeze(1)
                stats[k] = new_v.gather(1, perm0.to(device=new_v.device)).unbind(1)
        else:
            loss = torch.tensor(
                [
                    torch.tensor(
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
