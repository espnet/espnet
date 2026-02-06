from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.spk.loss.abs_loss import AbsLoss


class Softmax(AbsLoss):
    """Softmax loss

    Args:
        nout: Dimension of input features (embedding size)
        nclasses: Number of output classes
    """

    def __init__(self, nout: int, nclasses: int):
        super().__init__(nout)

        self.in_feats = nout

        self.weight = torch.nn.Parameter(
            torch.FloatTensor(nclasses, nout), requires_grad=True
        )
        nn.init.xavier_normal_(self.weight, gain=1)

        self.ce = nn.CrossEntropyLoss()

    def forward(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        r"""Forward pass of Softmax loss.

        Args:
            input: Input embeddings, shape (batch_size, embedding_dim)
            label: Ground truth labels, shape (batch_size,)

        Returns:
            loss: Cross-entropy loss
            accuracy: Classification accuracy
            preds: Predicted class indices
        """

        logits = F.linear(F.normalize(input), F.normalize(self.weight))
        preds = torch.argmax(logits, dim=1)

        if label is not None:
            if len(label.size()) == 2:
                label = label.squeeze(1)
            accuracy = (preds == label).float().mean()
        else:
            # Inference mode: return prediction only
            loss = None
            accuracy = None
            return loss, accuracy, preds

        loss = self.ce(logits, label)

        return loss, accuracy, preds
