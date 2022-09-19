import torch

from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet2.asvspoof.loss.abs_loss import AbsASVSpoofLoss


class ASVSpoofOCSoftmaxLoss(AbsASVSpoofLoss):
    """Binary loss for ASV Spoofing."""

    def __init__(
        self,
        weight: float = 1.0,
        enc_dim: int = 128,
        m_real: float = 0.5,
        m_fake: float = 0.2,
        alpha: float =20.0,
    ):
        super(ASVSpoofOCSoftmaxLoss).__init__()
        self.weight = weight
        self.feat_dim = enc_dim
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha
        self.center = torch.nn.Parameter(torch.randn(1, self.feat_dim))
        torch.nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = torch.nn.Softplus()

    def forward(self, label: torch.Tensor, emb: torch.Tensor, **kwargs):
        """Forward.
        Args:
            label (torch.Tensor): ground truth label [Batch, 1]
            emb   (torch.Tensor): encoder embedding output [Batch, T, enc_dim]
        """
        emb = torch.mean(emb, dim=1)
        w = torch.nn.functional.normalize(self.center, p=2, dim=1)
        x = torch.nn.functional.normalize(emb, p=2, dim=1)

        # TODO1 (exercise 2): compute scores based on w and x

        # TODO2 (exercise 2): calculate the score bias based on m_real and m_fake

        # TODO3 (exercise 2): apply alpha and softplus

        # TODO4 (exercise 2): returnthe final loss
        return None
    
    def score(self, emb: torch.Tensor):
        """Prediction.
        Args:
            emb (torch.Tensor): encoder embedding output [Batch, T, enc_dim]
        """
        emb = torch.mean(emb, dim=1)
        w = torch.nn.functional.normalize(self.center, p=2, dim=1)
        x = torch.nn.functional.normalize(emb, p=2, dim=1)

        # TODO5 (exercise 2): compute scores