"""Linear classifier with cross-entropy (no normalization, no margin)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from espnet2.spk.loss.abs_loss import AbsLoss


class LinearSoftmax(AbsLoss):
    """Linear layer (with bias) over the embedding, softmax cross-entropy.

    Unlike ``Softmax`` in ``espnet2/spk/loss/softmax.py``, which scores the
    cosine between the normalized embedding and class weights without a scale
    (logits in [-1, 1], so the probability of the right class stays small when
    there are many classes), the logits are unbounded, as in a plain
    classifier.

    Args:
        nout: Dimension of input features (embedding size).
        nclasses: Number of output classes.

    Example:
        >>> loss_fn = LinearSoftmax(nout=256, nclasses=107)
        >>> loss, accuracy, preds = loss_fn(embeddings, labels)
        >>> _, _, preds = loss_fn(embeddings)  # inference
    """

    def __init__(self, nout: int, nclasses: int):
        """Build the linear classifier."""
        super().__init__(nout)
        self.in_feats = nout
        self.linear = nn.Linear(nout, nclasses)
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Return the loss, the accuracy and the predicted classes.

        Args:
            input: Embeddings (batch, nout).
            label: Class indices (batch,) or (batch, 1); None at inference.

        Returns:
            Loss and accuracy (None without labels), and predicted classes.
        """
        logits = self.linear(input)
        preds = torch.argmax(logits, dim=1)
        if label is None:
            return None, None, preds
        if label.dim() == 2:
            label = label.squeeze(1)
        accuracy = (preds == label).float().mean()
        return self.ce(logits, label), accuracy, preds
