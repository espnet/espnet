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
            # Pooling over unpadded frames
            mean = torch.stack(
                [
                    torch.mean(x[i, :, : int(l.item())], dim=-1, keepdim=True)
                    for i, l in enumerate(feat_lengths)
                ],
                dim=0,
            ).repeat(1, 1, T)
            var = torch.stack(
                [
                    torch.var(
                        x[i, :, : int(l.item())], dim=-1, unbiased=False, keepdim=True
                    )
                    for i, l in enumerate(feat_lengths)
                ],
                dim=0,
            )
            var = var.clamp(min=1e-4, max=1e4)
            std = torch.sqrt(var).repeat(1, 1, T)
            global_x = torch.cat((x, mean, std), dim=1)
        else:
            global_x = torch.cat(
                (
                    x,
                    torch.mean(x, dim=2, keepdim=True).repeat(1, 1, T),
                    torch.sqrt(
                        torch.var(x, dim=2, keepdim=True, unbiased=False).clamp(
                            min=1e-4, max=1e4
                        )
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
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4))

        x = torch.cat((mu, sg), dim=1)

        return x
