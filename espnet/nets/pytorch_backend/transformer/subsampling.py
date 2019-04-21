import torch

from .embedding import PositionalEncoding


class Conv2dSubsampling(torch.nn.Module):
    def __init__(self, idim, dim, dropout):
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, dim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim, dim, 3, 2),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(dim * (idim // 4), dim),
            PositionalEncoding(dim, dropout)
        )

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]
