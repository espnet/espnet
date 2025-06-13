from itertools import permutations
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

# NOTE: this is copied here from Asteroid toolkit !
# All credits to Manuel Pariente:
# https://github.com/asteroid-team/asteroid/blob/master/asteroid/losses/pit_wrapper.py

class PITLossWrapper(nn.Module):
    r"""Permutation invariant loss wrapper.

    Args:
        loss_func: function with signature (est_targets, targets, **kwargs).
        pit_from (str): Determines how PIT is applied.

            * ``'pw_mtx'`` (pairwise matrix): `loss_func` computes pairwise
              losses and returns a torch.Tensor of shape
              :math:`(batch, n\_src, n\_src)`. Each element
              :math:`(batch, i, j)` corresponds to the loss between
              :math:`targets[:, i]` and :math:`est\_targets[:, j]`
            * ``'pw_pt'`` (pairwise point): `loss_func` computes the loss for
              a batch of single source and single estimates (tensors won't
              have the source axis). Output shape : :math:`(batch)`.
              See :meth:`~PITLossWrapper.get_pw_losses`.
            * ``'perm_avg'`` (permutation average): `loss_func` computes the
              average loss for a given permutations of the sources and
              estimates. Output shape : :math:`(batch)`.
              See :meth:`~PITLossWrapper.best_perm_from_perm_avg_loss`.

            In terms of efficiency, ``'perm_avg'`` is the least efficicient.

        perm_reduce (Callable): torch function to reduce permutation losses.
            Defaults to None (equivalent to mean). Signature of the func
            (pwl_set, **kwargs) : :math:`(B, n\_src!, n\_src) --> (B, n\_src!)`.
            `perm_reduce` can receive **kwargs during forward using the
            `reduce_kwargs` argument (dict). If those argument are static,
            consider defining a small function or using `functools.partial`.
            Only used in `'pw_mtx'` and `'pw_pt'` `pit_from` modes.

    For each of these modes, the best permutation and reordering will be
    automatically computed. When either ``'pw_mtx'`` or ``'pw_pt'`` is used,
    and the number of sources is larger than three, the hungarian algorithm is
    used to find the best permutation.

    Examples
        >>> import torch
        >>> from asteroid.losses import pairwise_neg_sisdr
        >>> sources = torch.randn(10, 3, 16000)
        >>> est_sources = torch.randn(10, 3, 16000)
        >>> # Compute PIT loss based on pairwise losses
        >>> loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
        >>> loss_val = loss_func(est_sources, sources)
        >>>
        >>> # Using reduce
        >>> def reduce(perm_loss, src):
        >>>     weighted = perm_loss * src.norm(dim=-1, keepdim=True)
        >>>     return torch.mean(weighted, dim=-1)
        >>>
        >>> loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx',
        >>>                            perm_reduce=reduce)
        >>> reduce_kwargs = {'src': sources}
        >>> loss_val = loss_func(est_sources, sources,
        >>>                      reduce_kwargs=reduce_kwargs)
    """

    def __init__(self, loss_func, pit_from="pw_mtx", perm_reduce=None):
        super().__init__()
        self.loss_func = loss_func
        self.pit_from = pit_from
        self.perm_reduce = perm_reduce
        if self.pit_from not in ["pw_mtx", "pw_pt", "perm_avg"]:
            raise ValueError(
                "Unsupported loss function type for now. Expected"
                "one of [`pw_mtx`, `pw_pt`, `perm_avg`]"
            )

    def forward(self, est_targets, targets, return_est=False, reduce_kwargs=None, **kwargs):
        r"""Find the best permutation and return the loss.

        Args:
            est_targets: torch.Tensor. Expected shape $(batch, nsrc, ...)$.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape $(batch, nsrc, ...)$.
                The batch of training targets
            return_est: Boolean. Whether to return the reordered targets
                estimates (To compute metrics or to save example).
            reduce_kwargs (dict or None): kwargs that will be passed to the
                pairwise losses reduce function (`perm_reduce`).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - Best permutation loss for each batch sample, average over
              the batch.
            - The reordered targets estimates if ``return_est`` is True.
              :class:`torch.Tensor` of shape $(batch, nsrc, ...)$.
        """
        n_src = targets.shape[1]
        assert n_src < 10, f"Expected source axis along dim 1, found {n_src}"
        if self.pit_from == "pw_mtx":
            # Loss function already returns pairwise losses
            pw_losses = self.loss_func(est_targets, targets, **kwargs)
        elif self.pit_from == "pw_pt":
            # Compute pairwise losses with a for loop.
            pw_losses = self.get_pw_losses(self.loss_func, est_targets, targets, **kwargs)
        elif self.pit_from == "perm_avg":
            # Cannot get pairwise losses from this type of loss.
            # Find best permutation directly.
            min_loss, batch_indices = self.best_perm_from_perm_avg_loss(
                self.loss_func, est_targets, targets, **kwargs
            )
            if not return_est:
                return min_loss
            return min_loss, batch_indices.to(min_loss.device)
        else:
            return

        assert pw_losses.ndim == 3, (
            "Something went wrong with the loss " "function, please read the docs."
        )
        assert pw_losses.shape[0] == targets.shape[0], "PIT loss needs same batch dim as input"

        reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()
        min_loss, batch_indices = self.find_best_perm(
            pw_losses, perm_reduce=self.perm_reduce, **reduce_kwargs
        )
        mean_loss = torch.mean(min_loss)
        if not return_est:
            return mean_loss
        reordered = self.reorder_source(est_targets, batch_indices)
        return mean_loss, reordered.to(mean_loss.device)

    @staticmethod
    def get_pw_losses(loss_func, est_targets, targets, **kwargs):
        r"""Get pair-wise losses between the training targets and its estimate
        for a given loss function.

        Args:
            loss_func: function with signature (est_targets, targets, **kwargs)
                The loss function to get pair-wise losses from.
            est_targets: torch.Tensor. Expected shape $(batch, nsrc, ...)$.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape $(batch, nsrc, ...)$.
                The batch of training targets.
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            torch.Tensor or size $(batch, nsrc, nsrc)$, losses computed for
            all permutations of the targets and est_targets.

        This function can be called on a loss function which returns a tensor
        of size :math:`(batch)`. There are more efficient ways to compute pair-wise
        losses using broadcasting.
        """
        batch_size, n_src, *_ = targets.shape
        pair_wise_losses = targets.new_empty(batch_size, n_src, n_src)
        for est_idx, est_src in enumerate(est_targets.transpose(0, 1)):
            for target_idx, target_src in enumerate(targets.transpose(0, 1)):
                pair_wise_losses[:, est_idx, target_idx] = loss_func(est_src, target_src, **kwargs)
        return pair_wise_losses

    @staticmethod
    def best_perm_from_perm_avg_loss(loss_func, est_targets, targets, **kwargs):
        r"""Find best permutation from loss function with source axis.

        Args:
            loss_func: function with signature $(est_targets, targets, **kwargs)$
                The loss function batch losses from.
            est_targets: torch.Tensor. Expected shape $(batch, nsrc, *)$.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape $(batch, nsrc, *)$.
                The batch of training targets.
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - :class:`torch.Tensor`:
                The loss corresponding to the best permutation of size $(batch,)$.

            - :class:`torch.Tensor`:
                The indices of the best permutations.
        """
        n_src = targets.shape[1]
        perms = torch.tensor(list(permutations(range(n_src))), dtype=torch.long)
        loss_set = torch.stack(
            [loss_func(est_targets[:, perm], targets, **kwargs) for perm in perms], dim=1
        )
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)
        # Permutation indices for each batch.
        batch_indices = torch.stack([perms[m] for m in min_loss_idx], dim=0)
        return min_loss, batch_indices

    @staticmethod
    def find_best_perm(pair_wise_losses, perm_reduce=None, **kwargs):
        r"""Find the best permutation, given the pair-wise losses.

        Dispatch between factorial method if number of sources is small (<3)
        and hungarian method for more sources. If ``perm_reduce`` is not None,
        the factorial method is always used.

        Args:
            pair_wise_losses (:class:`torch.Tensor`):
                Tensor of shape :math:`(batch, n\_src, n\_src)`. Pairwise losses.
            perm_reduce (Callable): torch function to reduce permutation losses.
                Defaults to None (equivalent to mean). Signature of the func
                (pwl_set, **kwargs) : :math:`(B, n\_src!, n\_src) -> (B, n\_src!)`
            **kwargs: additional keyword argument that will be passed to the
                permutation reduce function.

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size $(batch,)$.

            - :class:`torch.Tensor`:
              The indices of the best permutations.
        """
        n_src = pair_wise_losses.shape[-1]
        if perm_reduce is not None or n_src <= 3:
            min_loss, batch_indices = PITLossWrapper.find_best_perm_factorial(
                pair_wise_losses, perm_reduce=perm_reduce, **kwargs
            )
        else:
            min_loss, batch_indices = PITLossWrapper.find_best_perm_hungarian(pair_wise_losses)
        return min_loss, batch_indices

    @staticmethod
    def reorder_source(source, batch_indices):
        r"""Reorder sources according to the best permutation.

        Args:
            source (torch.Tensor): Tensor of shape :math:`(batch, n_src, time)`
            batch_indices (torch.Tensor): Tensor of shape :math:`(batch, n_src)`.
                Contains optimal permutation indices for each batch.

        Returns:
            :class:`torch.Tensor`: Reordered sources.
        """
        reordered_sources = torch.stack(
            [torch.index_select(s, 0, b) for s, b in zip(source, batch_indices)]
        )
        return reordered_sources

    @staticmethod
    def find_best_perm_factorial(pair_wise_losses, perm_reduce=None, **kwargs):
        r"""Find the best permutation given the pair-wise losses by looping
        through all the permutations.

        Args:
            pair_wise_losses (:class:`torch.Tensor`):
                Tensor of shape :math:`(batch, n_src, n_src)`. Pairwise losses.
            perm_reduce (Callable): torch function to reduce permutation losses.
                Defaults to None (equivalent to mean). Signature of the func
                (pwl_set, **kwargs) : :math:`(B, n\_src!, n\_src) -> (B, n\_src!)`
            **kwargs: additional keyword argument that will be passed to the
                permutation reduce function.

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size $(batch,)$.

            - :class:`torch.Tensor`:
              The indices of the best permutations.

        MIT Copyright (c) 2018 Kaituo XU.
        See `Original code
        <https://github.com/kaituoxu/Conv-TasNet/blob/master>`__ and `License
        <https://github.com/kaituoxu/Conv-TasNet/blob/master/LICENSE>`__.
        """
        n_src = pair_wise_losses.shape[-1]
        # After transposition, dim 1 corresp. to sources and dim 2 to estimates
        pwl = pair_wise_losses.transpose(-1, -2)
        perms = pwl.new_tensor(list(permutations(range(n_src))), dtype=torch.long)
        # Column permutation indices
        idx = torch.unsqueeze(perms, 2)
        # Loss mean of each permutation
        if perm_reduce is None:
            # one-hot, [n_src!, n_src, n_src]
            perms_one_hot = pwl.new_zeros((*perms.size(), n_src)).scatter_(2, idx, 1)
            loss_set = torch.einsum("bij,pij->bp", [pwl, perms_one_hot])
            loss_set /= n_src
        else:
            # batch = pwl.shape[0]; n_perm = idx.shape[0]
            # [batch, n_src!, n_src] : Pairwise losses for each permutation.
            pwl_set = pwl[:, torch.arange(n_src), idx.squeeze(-1)]
            # Apply reduce [batch, n_src!, n_src] --> [batch, n_src!]
            loss_set = perm_reduce(pwl_set, **kwargs)
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)

        # Permutation indices for each batch.
        batch_indices = torch.stack([perms[m] for m in min_loss_idx], dim=0)
        return min_loss, batch_indices

    @staticmethod
    def find_best_perm_hungarian(pair_wise_losses: torch.Tensor):
        """
        Find the best permutation given the pair-wise losses, using the Hungarian algorithm.

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size (batch,).

            - :class:`torch.Tensor`:
              The indices of the best permutations.
        """
        # After transposition, dim 1 corresp. to sources and dim 2 to estimates
        pwl = pair_wise_losses.transpose(-1, -2)
        # Just bring the numbers to cpu(), not the graph
        pwl_copy = pwl.detach().cpu()
        # Loop over batch + row indices are always ordered for square matrices.
        batch_indices = torch.tensor([linear_sum_assignment(pwl)[1] for pwl in pwl_copy]).to(
            pwl.device
        )
        min_loss = torch.gather(pwl, 2, batch_indices[..., None]).mean([-1, -2])
        return min_loss, batch_indices


class PITReorder(PITLossWrapper):
    """Permutation invariant reorderer. Only returns the reordered estimates.
    See `:py:class:asteroid.losses.PITLossWrapper`."""

    def forward(self, est_targets, targets, reduce_kwargs=None, **kwargs):
        _, reordered = super().forward(
            est_targets=est_targets,
            targets=targets,
            return_est=True,
            reduce_kwargs=reduce_kwargs,
            **kwargs,
        )
        return reordered