import torch

from espnet2.asvspoof.loss.abs_loss import AbsASVSpoofLoss
from espnet.nets.pytorch_backend.nets_utils import to_device


class ASVSpoofAMSoftmaxLoss(AbsASVSpoofLoss):
    """Binary loss for ASV Spoofing."""

    def __init__(
        self,
        weight: float = 1.0,
        enc_dim: int = 128,
        s: float = 20,
        m: float = 0.5,
    ):
        super(ASVSpoofAMSoftmaxLoss).__init__()
        self.weight = weight
        self.enc_dim = enc_dim
        self.s = s
        self.m = m
        self.centers = torch.nn.Parameter(torch.randn(2, enc_dim))
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = torch.nn.BCELoss(reduction="mean")

    def forward(self, label: torch.Tensor, emb: torch.Tensor, **kwargs):
        """Forward.
        Args:
            label (torch.Tensor): ground truth label [Batch, 1]
            emb   (torch.Tensor): encoder embedding output [Batch, T, enc_dim]
        """
        batch_size = emb.shape[0]
        emb = torch.mean(emb, dim=1)
        norms = torch.norm(emb, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(emb, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, 2)
        y_onehot.zero_()
        y_onehot = torch.autograd.Variable(y_onehot)
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)
        loss = self.loss(self.sigmoid(margin_logits[:, 0]), label.view(-1).float())

        return loss

    def score(self, emb: torch.Tensor):
        """Prediction.
        Args:
            emb (torch.Tensor): encoder embedding output [Batch, T, enc_dim]
        """
        emb = torch.mean(emb, dim=1)
        norms = torch.norm(emb, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(emb, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))
        return logits[:, 0]
