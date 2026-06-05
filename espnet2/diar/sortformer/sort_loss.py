"""Sortformer training targets and BCE loss (NeMo-free port).

Implements the two target-generation strategies and the masked binary
cross-entropy used to train Sortformer:

* ``get_ats_targets``  -- Arrival-Time-Sort (ATS) targets: speaker channels are
  ordered by their first-active frame and the best matching arrival-time-ordered
  permutation is selected against the predictions.
* ``get_pil_targets``  -- Permutation-Invariant-Loss (PIL) targets: the best of
  all speaker permutations is selected against the predictions.
* ``SortformerBCELoss`` -- BCE on sigmoid probabilities, averaged over valid
  (un-padded) frames and speakers.

The combined training loss is ``ats_weight * BCE(preds, ats_targets) +
pil_weight * BCE(preds, pil_targets)``.

Reference (Apache-2.0): NVIDIA/NeMo
    nemo/collections/asr/parts/utils/asr_multispeaker_utils.py
    nemo/collections/asr/losses/bce_loss.py
"""

import itertools

import torch
import torch.nn as nn


def find_first_nonzero(
    mat: torch.Tensor, max_cap_val: int = -1, thres: float = 0.5
) -> torch.Tensor:
    """First frame index where each speaker becomes active (``max_cap_val`` if never)."""
    labels_discrete = mat.clone()
    labels_discrete[labels_discrete < thres] = 0
    labels_discrete[labels_discrete >= thres] = 1
    non_zero_mask = labels_discrete != 0
    mask_max_values, mask_max_indices = torch.max(non_zero_mask, dim=1)
    mask_max_indices[mask_max_values == 0] = max_cap_val
    return mask_max_indices


def find_best_permutation(
    match_score: torch.Tensor, speaker_permutations: torch.Tensor
) -> torch.Tensor:
    """Pick, per batch item, the permutation with the highest match score."""
    batch_best_perm = torch.argmax(match_score, axis=1)
    rep = speaker_permutations.repeat(batch_best_perm.shape[0], 1).to(
        match_score.device
    )
    perm_size = speaker_permutations.shape[0]
    global_inds = (
        torch.arange(0, perm_size * batch_best_perm.shape[0], perm_size).to(
            batch_best_perm.device
        )
        + batch_best_perm
    )
    return rep[global_inds.to(rep.device), :]


def reconstruct_labels(
    labels: torch.Tensor, batch_perm_inds: torch.Tensor
) -> torch.Tensor:
    """Reorder the speaker axis of ``labels`` by the chosen permutation indices."""
    batch_size, num_frames, num_speakers = labels.shape
    batch_perm_inds_exp = batch_perm_inds.unsqueeze(1).expand(-1, num_frames, -1)
    return torch.gather(labels, 2, batch_perm_inds_exp)


def get_ats_targets(
    labels: torch.Tensor,
    preds: torch.Tensor,
    speaker_permutations: torch.Tensor,
    thres: float = 0.5,
    tolerance: float = 0,
) -> torch.Tensor:
    """Arrival-Time-Sort targets. Shapes: (B, T, S)."""
    nonzero_ind = find_first_nonzero(labels, max_cap_val=labels.shape[1], thres=thres)
    sorted_values = torch.sort(nonzero_ind)[0]
    perm_size = speaker_permutations.shape[0]
    permed_labels = labels[:, :, speaker_permutations]
    permed_nonzero_ind = find_first_nonzero(permed_labels, max_cap_val=labels.shape[1])
    perm_compare = (
        torch.abs(sorted_values.unsqueeze(1) - permed_nonzero_ind) <= tolerance
    )
    perm_mask = torch.all(perm_compare, dim=2).float()
    preds_rep = torch.unsqueeze(preds, 2).repeat(1, 1, perm_size, 1)
    match_score = torch.sum(permed_labels * preds_rep, axis=1).sum(axis=2) * perm_mask
    batch_perm_inds = find_best_permutation(match_score, speaker_permutations)
    return reconstruct_labels(labels, batch_perm_inds)


def get_pil_targets(
    labels: torch.Tensor, preds: torch.Tensor, speaker_permutations: torch.Tensor
) -> torch.Tensor:
    """Permutation-Invariant targets. Shapes: (B, T, S)."""
    permed_labels = labels[:, :, speaker_permutations]
    preds_rep = torch.unsqueeze(preds, 2).repeat(1, 1, speaker_permutations.shape[0], 1)
    match_score = torch.sum(permed_labels * preds_rep, axis=1).sum(axis=2)
    batch_perm_inds = find_best_permutation(match_score, speaker_permutations)
    return reconstruct_labels(labels, batch_perm_inds)


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
    """``ats_weight * BCE(ats) + pil_weight * BCE(pil)`` over all speaker permutations."""

    def __init__(
        self, num_spks: int = 4, ats_weight: float = 0.5, pil_weight: float = 0.5
    ):
        super().__init__()
        self.ats_weight = ats_weight
        self.pil_weight = pil_weight
        self.bce = SortformerBCELoss()
        perms = torch.tensor(list(itertools.permutations(range(num_spks))))
        self.register_buffer("speaker_permutations", perms, persistent=False)

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, target_lens: torch.Tensor
    ):
        """preds/targets: (B, T, S) -- preds are sigmoid probabilities.

        Returns (loss, ats_loss, pil_loss).
        """
        perms = self.speaker_permutations.to(preds.device)
        targets_ats = get_ats_targets(
            targets.clone(), preds, speaker_permutations=perms
        )
        targets_pil = get_pil_targets(
            targets.clone(), preds, speaker_permutations=perms
        )
        ats_loss = self.bce(preds, targets_ats, target_lens)
        pil_loss = self.bce(preds, targets_pil, target_lens)
        loss = self.ats_weight * ats_loss + self.pil_weight * pil_loss
        return loss, ats_loss, pil_loss
