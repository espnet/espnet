"""Powerset encoding for multi-speaker diarization.

Based on:
- pyannote.audio powerset implementation
- DiariZen segmentation approach

Powerset encoding converts multi-label speaker classification to multi-class
classification over all possible speaker combinations.
"""

from itertools import combinations
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Powerset(nn.Module):
    """Powerset encoding for speaker diarization.

    Converts between multi-label representation (each speaker independently)
    and powerset representation (all possible speaker combinations).

    Args:
        num_classes: Maximum number of speakers
        max_set_size: Maximum number of simultaneous speakers
            (e.g., max_set_size=2 allows up to 2 overlapping speakers)

    Example:
        For num_classes=3, max_set_size=2:
        Powerset classes: {}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}
        Total: 7 classes (1 + 3 + 3)
    """

    def __init__(
        self,
        num_classes: int,
        max_set_size: Optional[int] = None,
    ):
        super().__init__()

        self.num_classes = num_classes

        if max_set_size is None:
            max_set_size = num_classes
        self.max_set_size = max_set_size

        # Generate all possible speaker combinations
        self.mapping = self._build_mapping()
        self.num_powerset_classes = len(self.mapping)

        # Create conversion matrices
        # mapping_matrix: (num_powerset_classes, num_classes)
        # Each row indicates which speakers are active for that powerset class
        mapping_matrix = np.zeros(
            (self.num_powerset_classes, num_classes),
            dtype=np.float32
        )

        for k, val in enumerate(self.mapping):
            for v in val:
                mapping_matrix[k, v] = 1.0

        # Register as buffer (will be moved to GPU automatically)
        self.register_buffer(
            "mapping_matrix",
            torch.from_numpy(mapping_matrix)
        )

    def _build_mapping(self):
        """Build mapping from powerset index to speaker sets.

        Returns:
            List of tuples, where each tuple contains speaker indices
            E.g., for num_classes=3, max_set_size=2:
            [(), (0,), (1,), (2,), (0,1), (0,2), (1,2)]
        """
        mapping = [()]  # Empty set (no speakers)

        for set_size in range(1, self.max_set_size + 1):
            for speakers in combinations(range(self.num_classes), set_size):
                mapping.append(speakers)

        return mapping

    def to_multilabel(
        self,
        powerset: torch.Tensor
    ) -> torch.Tensor:
        """Convert powerset logits/predictions to multi-label format.

        Args:
            powerset: Powerset logits or predictions
                Shape: (batch, frames, num_powerset_classes) or
                       (batch, frames, num_powerset_classes) for probabilities

        Returns:
            Multi-label tensor of shape (batch, frames, num_classes)
            If input is logits, output is probabilities per speaker
            If input is hard predictions, output is binary labels
        """
        # Check if input is probabilities/logits or hard labels
        is_probabilistic = powerset.dtype == torch.float

        if is_probabilistic:
            # Convert logits to probabilities
            if powerset.shape[-1] == self.num_powerset_classes:
                # Softmax over powerset classes
                powerset_probs = F.softmax(powerset, dim=-1)
            else:
                powerset_probs = powerset  # Already probabilities

            # Matrix multiplication: (B, T, P) x (P, C) -> (B, T, C)
            multilabel = torch.matmul(
                powerset_probs,
                self.mapping_matrix
            )
        else:
            # Hard assignment: use argmax to get powerset class
            powerset_class = torch.argmax(powerset, dim=-1)  # (batch, frames)
            # Index into mapping matrix
            multilabel = self.mapping_matrix[powerset_class]  # (batch, frames, num_classes)

        return multilabel

    def to_powerset(
        self,
        multilabel: torch.Tensor
    ) -> torch.Tensor:
        """Convert multi-label format to powerset labels (hard labels).

        Args:
            multilabel: Multi-label tensor of shape (batch, frames, num_classes)
                Binary values indicating active speakers

        Returns:
            Powerset class indices of shape (batch, frames)
        """
        batch_size, num_frames, num_classes = multilabel.shape
        assert num_classes == self.num_classes, \
            f"Expected {self.num_classes} classes, got {num_classes}"

        # Flatten for processing
        multilabel_flat = multilabel.reshape(-1, num_classes)  # (B*T, C)

        # Find matching powerset class for each frame
        # Compute distance between each frame and each powerset class
        # Distance = number of mismatches
        # Shape: (B*T, num_powerset_classes)
        distances = torch.cdist(
            multilabel_flat.float(),
            self.mapping_matrix.float(),
            p=1  # L1 distance
        )

        # Find closest powerset class
        powerset_flat = torch.argmin(distances, dim=-1)  # (B*T,)

        # Reshape back
        powerset = powerset_flat.reshape(batch_size, num_frames)

        return powerset

    def get_cardinality_weights(
        self,
        weight_type: str = "linear"
    ) -> torch.Tensor:
        """Get weights for each powerset class based on cardinality.

        Useful for weighted loss to handle class imbalance.
        Higher cardinality (more simultaneous speakers) can be weighted more
        since overlapping speech is typically rarer.

        Args:
            weight_type: "linear", "quadratic", or "uniform"

        Returns:
            Tensor of shape (num_powerset_classes,) with weights
        """
        cardinalities = torch.tensor(
            [len(s) for s in self.mapping],
            dtype=torch.float32,
            device=self.mapping_matrix.device
        )

        if weight_type == "linear":
            weights = 1.0 + cardinalities
        elif weight_type == "quadratic":
            weights = 1.0 + cardinalities ** 2
        elif weight_type == "uniform":
            weights = torch.ones_like(cardinalities)
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

        # Normalize
        weights = weights / weights.mean()

        return weights

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"max_set_size={self.max_set_size}, "
            f"num_powerset_classes={self.num_powerset_classes})"
        )


def test_powerset():
    """Test powerset encoding/decoding."""
    print("Testing Powerset encoding...")

    # Test case 1: num_classes=3, max_set_size=2
    powerset = Powerset(num_classes=3, max_set_size=2)
    print(f"\n{powerset}")
    print(f"Mapping: {powerset.mapping}")
    print(f"Mapping matrix:\n{powerset.mapping_matrix}")

    # Test multilabel -> powerset -> multilabel
    multilabel = torch.tensor([
        [[1, 0, 0],  # Speaker 0 only
         [0, 1, 0],  # Speaker 1 only
         [1, 1, 0],  # Speakers 0 and 1
         [0, 0, 0]]  # No speakers
    ], dtype=torch.float32)

    print(f"\nInput multilabel:\n{multilabel}")

    powerset_labels = powerset.to_powerset(multilabel)
    print(f"\nPowerset labels:\n{powerset_labels}")

    recovered_multilabel = powerset.to_multilabel(
        F.one_hot(powerset_labels, num_classes=powerset.num_powerset_classes).float()
    )
    print(f"\nRecovered multilabel:\n{recovered_multilabel}")

    # Check if recovery is correct
    assert torch.allclose(multilabel, recovered_multilabel), "Recovery failed!"
    print("\nTest passed!")


if __name__ == "__main__":
    test_powerset()
