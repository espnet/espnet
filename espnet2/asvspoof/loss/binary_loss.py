import torch

from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet2.asvspoof.loss.abs_loss import AbsASVSpoofLoss


class ASVSpoofBinaryLoss(AbsASVSpoofLoss):
    """Binary loss for ASV Spoofing."""

    def __init__(
        self,
        weight: float = 1.0,
    ):
        super().__init__()
        self.weight = weight
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = torch.nn.BCELoss(reduction="mean")

    def forward(self, pred: torch.Tensor, label: torch.Tensor):
        """Forward.
        Args:
            pred  (torch.Tensor): prediction probability [Batch, 2]
            label (torch.Tensor): ground truth label [Batch, 2]
        """
        loss = self.loss(self.sigmoid(pred.view(-1)), label.view(-1).float())
        return loss