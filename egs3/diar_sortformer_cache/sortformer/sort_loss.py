"""Sortformer training targets and BCE loss (NeMo-free port).

Two target-generation strategies plus a masked BCE on sigmoid probabilities:

* ``get_ats_targets``  -- Arrival-Time-Sort (ATS) targets: speaker channels are
  ordered by their first-active frame (a single permutation per sample, obtained
  by ``argsort`` -- no permutation enumeration).
* ``get_pil_targets``  -- Permutation-Invariant targets: the optimal speaker
  assignment between targets and predictions, found by the **Hungarian**
  algorithm on the per-speaker activity-overlap affinity. This is mathematically
  identical to brute-force PIL (both maximise the same affinity) but scales to
  many speakers (``S!`` enumeration is infeasible at 8 speakers: 8! = 40320).
* ``SortformerBCELoss`` -- BCE on probabilities, averaged over valid frames.

The permutation selection is non-differentiable (argmax/assignment), so gradients
never flow through it; we detach predictions when computing the assignment.

Reference (Apache-2.0): NVIDIA/NeMo
    nemo/collections/asr/parts/utils/asr_multispeaker_utils.py
    nemo/collections/asr/losses/bce_loss.py
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def find_first_nonzero(
    mat: torch.Tensor, max_cap_val: int = -1, thres: float = 0.5
) -> torch.Tensor:
    """First frame index where each speaker is active (``max_cap_val`` if never)."""
    labels_discrete = (mat >= thres).to(mat.dtype)
    non_zero_mask = labels_discrete != 0
    mask_max_values, mask_max_indices = torch.max(non_zero_mask, dim=1)
    mask_max_indices[mask_max_values == 0] = max_cap_val
    return mask_max_indices


def reconstruct_labels(
    labels: torch.Tensor, batch_perm_inds: torch.Tensor
) -> torch.Tensor:
    """Reorder the speaker axis: ``out[b, :, s] = labels[b, :, perm[b, s]]``."""
    batch_size, num_frames, num_speakers = labels.shape
    batch_perm_inds_exp = batch_perm_inds.unsqueeze(1).expand(-1, num_frames, -1)
    return torch.gather(labels, 2, batch_perm_inds_exp)


def get_ats_targets(labels: torch.Tensor, thres: float = 0.5) -> torch.Tensor:
    """Arrival-Time-Sort targets: speakers ordered by first-active frame.

    Shapes: ``labels`` (B, T, S) -> (B, T, S). Scales to any S (argsort).
    """
    nonzero_ind = find_first_nonzero(labels, max_cap_val=labels.shape[1], thres=thres)
    perm = torch.argsort(nonzero_ind, dim=1, stable=True)  # (B, S) arrival order
    return reconstruct_labels(labels, perm)


def get_pil_targets(labels: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    """Permutation-Invariant targets via Hungarian assignment.

    Shapes: (B, T, S). Aligns each prediction channel ``j`` to the target speaker
    ``i`` that maximises activity overlap ``sum_t labels[:,t,i] * preds[:,t,j]``.
    """
    L = labels.detach()
    P = preds.detach()
    affinity = torch.einsum("bti,btj->bij", L, P)  # (B, S_tgt, S_pred)
    affinity_np = affinity.cpu().numpy()
    s = labels.shape[2]
    perms = np.empty((labels.shape[0], s), dtype=np.int64)
    for b in range(labels.shape[0]):
        row, col = linear_sum_assignment(-affinity_np[b])  # maximise affinity
        perms[b, col] = row  # perm[pred j] = target speaker i
    perm = torch.from_numpy(perms).to(labels.device)
    return reconstruct_labels(labels, perm)


class SortformerBCELoss(nn.Module):
    """Masked BCE on sigmoid probabilities, averaged over valid frames & speakers."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.loss_f = nn.BCELoss(reduction=reduction)

    def forward(
        self, probs: torch.Tensor, labels: torch.Tensor, target_lens: torch.Tensor
    ) -> torch.Tensor:
        probs_list = [probs[k, : target_lens[k], :] for k in range(probs.shape[0])]
        labels_list = [labels[k, : target_lens[k], :] for k in range(labels.shape[0])]
        probs = torch.cat(probs_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        probs = probs.clamp(min=0.0, max=1.0)
        return self.loss_f(probs, labels)


class SortformerHybridLoss(nn.Module):
    """``ats_weight * BCE(ATS targets) + pil_weight * BCE(PIL targets)``.

    Scales to any number of speakers (Hungarian PIL + argsort ATS).
    """

    def __init__(
        self, num_spks: int = 4, ats_weight: float = 0.5, pil_weight: float = 0.5
    ):
        super().__init__()
        self.num_spks = num_spks
        self.ats_weight = ats_weight
        self.pil_weight = pil_weight
        self.bce = SortformerBCELoss()

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, target_lens: torch.Tensor
    ):
        """preds/targets: (B, T, S) -- preds are sigmoid probabilities.

        Returns ``(loss, ats_loss, pil_loss)``.
        """
        targets_ats = get_ats_targets(targets)
        targets_pil = get_pil_targets(targets, preds)
        ats_loss = self.bce(preds, targets_ats, target_lens)
        pil_loss = self.bce(preds, targets_pil, target_lens)
        loss = self.ats_weight * ats_loss + self.pil_weight * pil_loss
        return loss, ats_loss, pil_loss
