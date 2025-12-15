from typing import Optional

import torch
import torch.nn as nn

from espnet2.spk.pooling.abs_pooling import AbsPooling


class ChnAttnStatPooling(AbsPooling):
    """Aggregates frame-level features to single utterance-level feature.

    Reference:
    ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
    in TDNN Based Speaker Verification
    https://arxiv.org/pdf/2005.07143

    Args:
        input_size: Dimension of the input frame-level embeddings.
                    The output dimensionality will be 2 × input_size
                    after concatenating mean and std.
    """

    def __init__(self, input_size: int = 1536):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_size * 3, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, input_size, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=2)
        self._output_size = input_size * 2

    def output_size(self):
        return self._output_size

    def forward(
        self,
        x: torch.Tensor,
        feat_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Forward pass of channel-attentive statistical pooling.

        Args:
            x: Input feature tensor of shape (batch_size, feature_dim, seq_len)
            feat_lengths: Optional tensor of shape (batch_size,) containing
                          the valid length of each sequence before padding

        Returns:
            x: Utterance-level embeddings of shape (batch_size, 2 × feature_dim)
        """

        T = x.size(-1)
        if feat_lengths is not None:
            feat_lengths = feat_lengths.to(x.device)
            # Pooling over unpadded frames
            mask = (
                torch.arange(T, device=x.device)[None, None, :]
                < feat_lengths[:, None, None]
            )
            # set padding to 0 for sum
            masked_x = x.masked_fill(~mask, 0)
            # sum over time
            sum_val = masked_x.sum(dim=-1, keepdim=True)
            sum_sq_val = (masked_x**2).sum(dim=-1, keepdim=True)
            # feat_lengths might be 0, add epsilon
            feat_lengths_ = feat_lengths.view(-1, 1, 1).clamp(min=1)
            mean = sum_val / feat_lengths_
            # var = E[X^2] - (E[X])^2
            var = sum_sq_val / feat_lengths_ - mean**2
            # add max clamp to prevent gradient explosion
            std = torch.sqrt(var.clamp(min=torch.finfo(var.dtype).eps, max=1e4))
            # Repeat mean and std to match x's time dimension
            mean = mean.repeat(1, 1, T)
            std = std.repeat(1, 1, T)
            global_x = torch.cat((x, mean, std), dim=1)
        else:
            var = torch.var(x, dim=2, keepdim=True, unbiased=False)
            global_x = torch.cat(
                (
                    x,
                    torch.mean(x, dim=2, keepdim=True).repeat(1, 1, T),
                    torch.sqrt(
                        var.clamp(
                            min=torch.finfo(x.dtype).eps, max=1e4
                        )  # clamp max to prevent gradient explosion
                    ).repeat(1, 1, T),
                ),
                dim=1,
            )

        w = self.attention(global_x)
        if feat_lengths is not None:
            # Apply padding mask
            padding_mask = torch.arange(T).expand(x.size(0), T).to(
                x.device
            ) >= feat_lengths.unsqueeze(
                1
            )  # (batch_size, seq_len)
            w = w.masked_fill(padding_mask.unsqueeze(1), torch.finfo(w.dtype).min)
        w = self.softmax(w)

        mu = torch.sum(x * w, dim=2)
        # add max clamp to prevent gradient explosion
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4))

        x = torch.cat((mu, sg), dim=1)

        return x
