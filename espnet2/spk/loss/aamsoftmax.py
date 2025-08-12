# Code from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.spk.loss.abs_loss import AbsLoss


class AAMSoftmax(AbsLoss):
    """Additive angular margin softmax.

    Reference:
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    https://arxiv.org/pdf/1801.07698

    Args:
        nout: Dimension of input features (embedding size)
        nclasses: Number of output classes
        margin: Angular margin for positive samples
        scale: Feature scaling factor
        easy_margin: Whether to use easy margin variant
    """

    def __init__(
        self,
        nout: int,
        nclasses: int,
        margin: float = 0.3,
        scale: int = 15,
        easy_margin: bool = False,
        **kwargs,
    ):
        super().__init__(nout)

        self.m = margin
        self.s = scale
        self.in_feats = nout

        self.weight = torch.nn.Parameter(
            torch.FloatTensor(nclasses, nout), requires_grad=True
        )
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - margin)  # Threshold for easy_margin
        self.mm = math.sin(math.pi - margin) * margin  # For numerical stability

        self.ce = nn.CrossEntropyLoss()

    def forward(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        r"""Forward pass of AAMSoftmax loss.

        Args:
            input: Input embeddings, shape (batch_size, embedding_dim)
            label: Ground truth labels, shape (batch_size,)

        Returns:
            loss: Cross-entropy loss with angular margins
            accuracy: Classification accuracy
            preds: Predicted class indices
        """

        cosine = F.linear(
            F.normalize(input), F.normalize(self.weight)
        )  # Output: (batch_size, nclasses)

        preds = torch.argmax(cosine, dim=1)

        if label is not None:
            if len(label.size()) == 2:
                label = label.squeeze(1)
            accuracy = (preds == label).float().mean()
        else:
            # Inference mode: return prediction only
            loss = None
            accuracy = None
            return loss, accuracy, preds

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))

        # Apply angular margin to positive samples:
        # cos(θ + m) = cosθ·cos(m) - sinθ·sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Handle numerical stability for angular margin
        if self.easy_margin:
            # Easy margin: only apply margin when cosine > 0
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard margin with numerical stability
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = (one_hot * phi) + (  # Ground truth: apply positive margin
            (1.0 - one_hot) * cosine
        )  # Other negatives: no margin

        # Scale logits for stable training
        output = output * self.s

        loss = self.ce(output, label)
        return loss, accuracy, preds
