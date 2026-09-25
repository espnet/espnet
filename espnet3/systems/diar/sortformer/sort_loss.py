"""Sortformer training targets and BCE loss (NeMo-free port).

The Sortformer model emits one sigmoid activity probability per speaker channel
per frame. Because the channel order is arbitrary, training compares the
predictions against two re-ordered (sorted) views of the reference labels and
sums their BCE. This module provides those two target reorderings plus the masked
BCE loss:

* ``get_ats_targets`` -- Arrival-Time-Sort (ATS) targets: speaker channels are
  reordered by their first-active frame. One deterministic permutation per
  sample, obtained with ``argsort`` (no permutation enumeration).
* ``get_pil_targets`` -- Permutation-Invariant targets: the optimal speaker
  assignment between reference and prediction channels, found by the **Hungarian**
  algorithm on the per-speaker activity-overlap affinity. Mathematically
  identical to brute-force PIL (both maximise the same affinity) but scales to
  many speakers, where ``S!`` enumeration is infeasible (8! = 40320 at 8 spk).
* ``SortformerBCELoss`` -- BCE on probabilities, averaged over valid frames.
* ``SortformerHybridLoss`` -- the weighted sum ``ats_weight * ATS-BCE +
  pil_weight * PIL-BCE`` used to train Sortformer.

Permutation selection is non-differentiable (argmax / assignment), so gradients
never flow through it; predictions are detached when computing the PIL
assignment. All tensors use the layout ``(B, T, S)`` for batch, time frames, and
speaker channels.

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
    """Find the first active frame index per speaker.

    A speaker is "active" at a frame when its value is ``>= thres``. Speakers that
    are never active get ``max_cap_val`` so they sort last under arrival-time sort.

    Args:
        mat: Activity tensor of shape ``(B, T, S)``.
        max_cap_val: Index assigned to never-active speakers.
        thres: Activity threshold.

    Returns:
        Long tensor of shape ``(B, S)`` with the first-active frame per speaker.
    """
    labels_discrete = (mat >= thres).to(mat.dtype)
    non_zero_mask = labels_discrete != 0
    mask_max_values, mask_max_indices = torch.max(non_zero_mask, dim=1)
    mask_max_indices[mask_max_values == 0] = max_cap_val
    return mask_max_indices


def reconstruct_labels(
    labels: torch.Tensor, batch_perm_inds: torch.Tensor
) -> torch.Tensor:
    """Reorder the speaker axis of ``labels`` by a per-sample permutation.

    Computes ``out[b, :, s] = labels[b, :, perm[b, s]]``.

    Args:
        labels: Tensor of shape ``(B, T, S)``.
        batch_perm_inds: Long permutation indices of shape ``(B, S)``.

    Returns:
        Reordered tensor of shape ``(B, T, S)``.
    """
    batch_size, num_frames, num_speakers = labels.shape
    batch_perm_inds_exp = batch_perm_inds.unsqueeze(1).expand(-1, num_frames, -1)
    return torch.gather(labels, 2, batch_perm_inds_exp)


def get_ats_targets(labels: torch.Tensor, thres: float = 0.5) -> torch.Tensor:
    """Build Arrival-Time-Sort (ATS) targets.

    Reorders speaker channels so that the speaker who first becomes active appears
    in channel 0, the next in channel 1, and so on. Uses a single ``argsort`` per
    sample, so it scales to any number of speakers.

    Args:
        labels: Reference activity tensor of shape ``(B, T, S)``.
        thres: Activity threshold passed to ``find_first_nonzero``.

    Returns:
        Reordered labels of shape ``(B, T, S)``.
    """
    nonzero_ind = find_first_nonzero(labels, max_cap_val=labels.shape[1], thres=thres)
    perm = torch.argsort(nonzero_ind, dim=1, stable=True)  # (B, S) arrival order
    return reconstruct_labels(labels, perm)


def get_pil_targets(labels: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    """Build Permutation-Invariant (PIL) targets via Hungarian assignment.

    Reorders reference channels to best match the prediction channels. For each
    sample, the affinity between target speaker ``i`` and prediction channel ``j``
    is the activity overlap ``sum_t labels[:, t, i] * preds[:, t, j]``; the
    Hungarian algorithm picks the assignment that maximises total affinity. Both
    inputs are detached, so this never contributes gradients.

    Args:
        labels: Reference activity tensor of shape ``(B, T, S)``.
        preds: Predicted sigmoid probabilities of shape ``(B, T, S)``.

    Returns:
        Reference labels reordered to align with ``preds``, shape ``(B, T, S)``.
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
    """Masked binary cross-entropy on sigmoid probabilities.

    Frames beyond each sample's valid length are dropped before the BCE so that
    padding does not contribute to the loss. Probabilities are clamped to
    ``[0, 1]`` for numerical safety.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.loss_f = nn.BCELoss(reduction=reduction)

    def forward(
        self, probs: torch.Tensor, labels: torch.Tensor, target_lens: torch.Tensor
    ) -> torch.Tensor:
        """Compute the masked BCE.

        Args:
            probs: Predicted sigmoid probabilities of shape ``(B, T, S)``.
            labels: Target activities of shape ``(B, T, S)``.
            target_lens: Valid frame count per sample, shape ``(B,)``.

        Returns:
            Scalar loss averaged over all valid frames and speakers.
        """
        probs_list = [probs[k, : target_lens[k], :] for k in range(probs.shape[0])]
        labels_list = [labels[k, : target_lens[k], :] for k in range(labels.shape[0])]
        probs = torch.cat(probs_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        probs = probs.clamp(min=0.0, max=1.0)
        return self.loss_f(probs, labels)


class SortformerHybridLoss(nn.Module):
    """Sortformer hybrid loss: weighted ATS-BCE plus PIL-BCE.

    Computes ``ats_weight * BCE(ATS targets) + pil_weight * BCE(PIL targets)``.
    The ATS term encourages a stable arrival-time channel order while the PIL term
    keeps the loss permutation-invariant. Scales to any number of speakers
    (Hungarian PIL + argsort ATS).

    Args:
        num_spks: Number of speaker channels the model predicts.
        ats_weight: Weight of the arrival-time-sort BCE term.
        pil_weight: Weight of the permutation-invariant BCE term.

    Example:
        >>> loss_fn = SortformerHybridLoss(num_spks=4)
        >>> # preds, targets: (B, T, 4); target_lens: (B,)
        >>> loss, ats_loss, pil_loss = loss_fn(preds, targets, target_lens)
        >>> loss.backward()
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
        """Compute the hybrid loss and its two components.

        Args:
            preds: Predicted sigmoid probabilities of shape ``(B, T, S)``.
            targets: Reference activities of shape ``(B, T, S)``.
            target_lens: Valid frame count per sample, shape ``(B,)``.

        Returns:
            Tuple ``(loss, ats_loss, pil_loss)`` of scalar tensors, where
            ``loss = ats_weight * ats_loss + pil_weight * pil_loss``.
        """
        targets_ats = get_ats_targets(targets)
        targets_pil = get_pil_targets(targets, preds)
        ats_loss = self.bce(preds, targets_ats, target_lens)
        pil_loss = self.bce(preds, targets_pil, target_lens)
        loss = self.ats_weight * ats_loss + self.pil_weight * pil_loss
        return loss, ats_loss, pil_loss
