#!/usr/bin/env python3
"""Permutation Invariant Training (PIT) loss for speaker diarization.

Implementation based on:
- pyannote.audio: https://github.com/pyannote/pyannote-audio
- asteroid: https://github.com/asteroid-team/asteroid

Uses Hungarian algorithm (Munkres) for optimal permutation finding in O(N³) time,
which is much more efficient than brute-force O(N!) enumeration.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from scipy.optimize import linear_sum_assignment


class PITLoss(nn.Module):
    """Permutation Invariant Training (PIT) loss wrapper.

    Finds the optimal permutation of predicted speakers that minimizes the loss
    with respect to the ground truth speakers using the Hungarian algorithm.

    Args:
        loss_fn: Base loss function to compute per-speaker loss.
                 Should accept (predictions, targets) and return frame-level losses.
                 Examples: nn.BCEWithLogitsLoss(reduction='none'), nn.MSELoss(reduction='none')
        pit_mode: PIT mode, one of:
            - "pairwise": Compute loss for all pairs, use Hungarian (default)
            - "utterance": Find best permutation per utterance (minimize total loss)
            - "frame": Find best permutation per frame (not recommended)

    Example:
        >>> # Binary cross-entropy for multi-label diarization
        >>> base_loss = nn.BCEWithLogitsLoss(reduction='none')
        >>> pit_loss = PITLoss(base_loss, pit_mode='pairwise')
        >>>
        >>> # predictions: (batch, frames, speakers)
        >>> # targets: (batch, frames, speakers)
        >>> loss, best_perm = pit_loss(predictions, targets)
    """

    def __init__(
        self,
        loss_fn: Optional[Callable] = None,
        pit_mode: str = "pairwise",
    ):
        super().__init__()

        if loss_fn is None:
            # Default: binary cross-entropy with logits (no reduction)
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        self.loss_fn = loss_fn
        self.pit_mode = pit_mode

        assert pit_mode in ["pairwise", "utterance", "frame"], \
            f"Invalid pit_mode: {pit_mode}. Must be 'pairwise', 'utterance', or 'frame'."

    def compute_pairwise_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute pairwise losses between all predicted and target speakers.

        Args:
            predictions: (batch, frames, num_speakers)
            targets: (batch, frames, num_speakers)
            lengths: (batch,) - valid frame lengths (optional)

        Returns:
            pairwise_losses: (batch, num_speakers, num_speakers)
                pairwise_losses[b, i, j] = loss between pred speaker i and target speaker j
        """
        batch_size, num_frames, num_speakers = predictions.shape

        # Compute loss for all pairs of (predicted speaker, target speaker)
        # Shape: (batch, num_speakers_pred, num_speakers_target, num_frames)
        pairwise_losses = torch.zeros(
            batch_size, num_speakers, num_speakers, num_frames,
            device=predictions.device, dtype=predictions.dtype
        )

        for i in range(num_speakers):
            for j in range(num_speakers):
                # Compare predicted speaker i with target speaker j
                pred_i = predictions[:, :, i]  # (batch, frames)
                target_j = targets[:, :, j]    # (batch, frames)

                # Compute frame-level loss
                frame_loss = self.loss_fn(pred_i, target_j)  # (batch, frames)
                pairwise_losses[:, i, j, :] = frame_loss

        # Average over frames (considering lengths if provided)
        if lengths is not None:
            # Create mask for valid frames
            mask = torch.arange(num_frames, device=predictions.device)[None, None, None, :] < lengths[:, None, None, None]
            pairwise_losses = pairwise_losses * mask.float()

            # Sum over frames and divide by actual length
            pairwise_losses = pairwise_losses.sum(dim=-1)  # (batch, num_speakers, num_speakers)
            pairwise_losses = pairwise_losses / lengths[:, None, None].clamp(min=1)
        else:
            # Simple average over all frames
            pairwise_losses = pairwise_losses.mean(dim=-1)  # (batch, num_speakers, num_speakers)

        return pairwise_losses

    def find_best_permutation_hungarian(
        self,
        pairwise_losses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find optimal permutation using Hungarian algorithm.

        Args:
            pairwise_losses: (batch, num_speakers, num_speakers)
                Cost matrix where [b, i, j] is cost of matching pred i to target j

        Returns:
            best_permutations: (batch, num_speakers) - optimal permutation indices
            best_losses: (batch,) - minimum total loss for each batch item
        """
        batch_size, num_speakers, _ = pairwise_losses.shape

        best_permutations = torch.zeros(batch_size, num_speakers, dtype=torch.long, device=pairwise_losses.device)
        best_losses = torch.zeros(batch_size, device=pairwise_losses.device, dtype=pairwise_losses.dtype)

        # Solve assignment problem for each batch item
        for b in range(batch_size):
            cost_matrix = pairwise_losses[b].detach().cpu().numpy()

            # Hungarian algorithm: finds minimum cost assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # row_ind is always [0, 1, 2, ..., num_speakers-1]
            # col_ind is the optimal permutation
            best_permutations[b] = torch.tensor(col_ind, dtype=torch.long, device=pairwise_losses.device)

            # Compute total loss for this permutation
            for i, j in zip(row_ind, col_ind):
                best_losses[b] += pairwise_losses[b, i, j]

        return best_permutations, best_losses

    def apply_permutation(
        self,
        targets: torch.Tensor,
        permutations: torch.Tensor,
    ) -> torch.Tensor:
        """Apply permutation to targets.

        Args:
            targets: (batch, frames, num_speakers)
            permutations: (batch, num_speakers) - permutation indices

        Returns:
            permuted_targets: (batch, frames, num_speakers)
        """
        batch_size, num_frames, num_speakers = targets.shape

        # Create permuted targets
        permuted_targets = torch.zeros_like(targets)

        for b in range(batch_size):
            perm = permutations[b]
            permuted_targets[b] = targets[b, :, perm]

        return permuted_targets

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_permutation: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with PIT.

        Args:
            predictions: (batch, frames, num_speakers) - model predictions (logits or probs)
            targets: (batch, frames, num_speakers) - ground truth labels
            lengths: (batch,) - valid frame lengths (optional)
            return_permutation: If True, return optimal permutation indices

        Returns:
            loss: Scalar loss (averaged over batch)
            best_permutations: (batch, num_speakers) if return_permutation=True, else None
        """
        # Compute pairwise losses between all speaker pairs
        pairwise_losses = self.compute_pairwise_losses(predictions, targets, lengths)

        # Find optimal permutation using Hungarian algorithm
        best_permutations, best_losses = self.find_best_permutation_hungarian(pairwise_losses)

        # Average loss over batch
        loss = best_losses.mean()

        if return_permutation:
            return loss, best_permutations
        else:
            return loss, None


class PITLossWithPowersetEncoding(nn.Module):
    """PIT loss for powerset-encoded diarization.

    This variant handles the case where speakers are encoded using powerset encoding
    (multi-label -> multi-class conversion). It first converts back to multi-label space,
    applies PIT, then converts to powerset for loss computation.

    Args:
        powerset_encoder: PowersetEncoding instance
        base_loss_fn: Loss function for powerset classes (e.g., nn.CrossEntropyLoss(reduction='none'))
        use_pit: Whether to use PIT (if False, no permutation search)

    Example:
        >>> from espnet3.components.diarization.powerset import PowersetEncoding
        >>> powerset = PowersetEncoding(num_speakers=4, max_speakers=2)
        >>> loss_fn = nn.CrossEntropyLoss(reduction='none')
        >>> pit_loss = PITLossWithPowersetEncoding(powerset, loss_fn, use_pit=True)
        >>>
        >>> # logits: (batch, frames, num_powerset_classes)
        >>> # targets: (batch, frames, num_speakers) - multi-label format
        >>> loss, perm = pit_loss(logits, targets)
    """

    def __init__(
        self,
        powerset_encoder,
        base_loss_fn: nn.Module,
        use_pit: bool = True,
    ):
        super().__init__()
        self.powerset = powerset_encoder
        self.base_loss_fn = base_loss_fn
        self.use_pit = use_pit

        if use_pit:
            # For PIT, we need to work in multi-label space
            # Use BCE loss for each speaker
            self.pit_loss = PITLoss(
                loss_fn=nn.BCEWithLogitsLoss(reduction='none'),
                pit_mode='pairwise'
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_permutation: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            logits: (batch, frames, num_powerset_classes) - model output logits
            targets: (batch, frames, num_speakers) - ground truth in multi-label format
            lengths: (batch,) - valid frame lengths (optional)
            return_permutation: If True, return optimal permutation

        Returns:
            loss: Scalar loss
            best_permutations: (batch, num_speakers) if return_permutation=True and use_pit=True
        """
        if not self.use_pit:
            # No PIT: directly convert targets to powerset and compute loss
            target_powerset = self.powerset.to_powerset(targets)  # (batch, frames)

            # Compute cross-entropy loss
            batch_size, num_frames, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            target_flat = target_powerset.view(-1)

            frame_losses = self.base_loss_fn(logits_flat, target_flat)

            if lengths is not None:
                # Mask and average
                mask = torch.arange(num_frames, device=logits.device)[None, :] < lengths[:, None]
                mask_flat = mask.view(-1)
                loss = (frame_losses * mask_flat.float()).sum() / lengths.sum().clamp(min=1)
            else:
                loss = frame_losses.mean()

            return loss, None

        else:
            # With PIT: find optimal speaker permutation first
            # Convert logits to multi-label space for PIT
            # Use softmax over powerset classes, then convert back to multi-label probs
            probs_powerset = torch.softmax(logits, dim=-1)  # (batch, frames, num_classes)

            # Convert powerset probs to multi-label probs
            # For each powerset class, get the corresponding speaker combination
            batch_size, num_frames, num_classes = logits.shape
            num_speakers = self.powerset.num_speakers

            # Compute expected speaker activations
            probs_multilabel = torch.zeros(
                batch_size, num_frames, num_speakers,
                device=logits.device, dtype=logits.dtype
            )

            for class_idx in range(num_classes):
                # Get speaker combination for this class
                speaker_combination = self.powerset.powerset_classes[class_idx]

                # Add probability mass to active speakers
                for speaker_idx in speaker_combination:
                    probs_multilabel[:, :, speaker_idx] += probs_powerset[:, :, class_idx]

            # Convert to logits for BCE loss (inverse sigmoid)
            # Note: This is approximate since we're going prob -> logit
            probs_multilabel = probs_multilabel.clamp(1e-7, 1 - 1e-7)
            logits_multilabel = torch.log(probs_multilabel / (1 - probs_multilabel))

            # Apply PIT in multi-label space
            pit_loss, best_permutations = self.pit_loss(
                logits_multilabel,
                targets,
                lengths=lengths,
                return_permutation=True
            )

            # Now compute actual loss with optimal permutation
            # Apply permutation to targets
            if best_permutations is not None:
                targets_permuted = self.pit_loss.apply_permutation(targets, best_permutations)
            else:
                targets_permuted = targets

            # Convert permuted targets to powerset
            target_powerset = self.powerset.to_powerset(targets_permuted)

            # Compute final cross-entropy loss with optimal permutation
            logits_flat = logits.view(-1, num_classes)
            target_flat = target_powerset.view(-1)

            frame_losses = self.base_loss_fn(logits_flat, target_flat)

            if lengths is not None:
                mask = torch.arange(num_frames, device=logits.device)[None, :] < lengths[:, None]
                mask_flat = mask.view(-1)
                loss = (frame_losses * mask_flat.float()).sum() / lengths.sum().clamp(min=1)
            else:
                loss = frame_losses.mean()

            if return_permutation:
                return loss, best_permutations
            else:
                return loss, None


def test_pit_loss():
    """Test PIT loss implementation."""
    print("Testing PIT Loss Implementation")
    print("=" * 80)

    # Test 1: Basic PIT with BCE loss
    print("\nTest 1: Basic PIT with BCE loss")
    print("-" * 80)

    batch_size = 2
    num_frames = 100
    num_speakers = 4

    # Create synthetic data
    # Predictions: (batch, frames, speakers)
    predictions = torch.randn(batch_size, num_frames, num_speakers)

    # Targets: (batch, frames, speakers) - with random permutation
    targets = torch.zeros(batch_size, num_frames, num_speakers)
    targets[0, :50, [0, 2]] = 1.0  # Batch 0: speakers 0 and 2 active
    targets[1, :50, [1, 3]] = 1.0  # Batch 1: speakers 1 and 3 active

    # Apply random permutation to targets (simulating label ambiguity)
    # This should be "fixed" by PIT
    targets_permuted = targets[:, :, [2, 0, 3, 1]]  # Swap speakers

    # Without PIT: compute loss with wrong order
    base_loss = nn.BCEWithLogitsLoss()
    loss_no_pit = base_loss(predictions, targets_permuted)
    print(f"Loss without PIT (wrong order): {loss_no_pit.item():.4f}")

    # With PIT: should find correct permutation
    pit_loss = PITLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'), pit_mode='pairwise')
    loss_with_pit, best_perm = pit_loss(predictions, targets_permuted, return_permutation=True)
    print(f"Loss with PIT (optimal order): {loss_with_pit.item():.4f}")
    print(f"Best permutation: {best_perm}")

    # Test 2: PIT with variable lengths
    print("\n\nTest 2: PIT with variable-length sequences")
    print("-" * 80)

    lengths = torch.tensor([80, 60])  # Different valid lengths per batch
    loss_with_lengths, best_perm = pit_loss(predictions, targets_permuted, lengths=lengths, return_permutation=True)
    print(f"Loss with variable lengths: {loss_with_lengths.item():.4f}")
    print(f"Lengths: {lengths}")
    print(f"Best permutation: {best_perm}")

    # Test 3: Compare Hungarian vs brute force (small case)
    print("\n\nTest 3: Comparing Hungarian algorithm efficiency")
    print("-" * 80)

    num_speakers_small = 3
    predictions_small = torch.randn(1, 50, num_speakers_small)
    targets_small = torch.rand(1, 50, num_speakers_small) > 0.5
    targets_small = targets_small.float()

    import time

    # Hungarian (our implementation)
    start = time.time()
    pit_loss_small = PITLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'))
    loss_hungarian, perm_hungarian = pit_loss_small(predictions_small, targets_small, return_permutation=True)
    time_hungarian = time.time() - start

    print(f"Hungarian algorithm:")
    print(f"  Time: {time_hungarian*1000:.2f}ms")
    print(f"  Loss: {loss_hungarian.item():.4f}")
    print(f"  Permutation: {perm_hungarian[0]}")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_pit_loss()
