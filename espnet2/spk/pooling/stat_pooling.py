from typing import Optional

import torch

from espnet2.spk.pooling.abs_pooling import AbsPooling


class StatsPooling(AbsPooling):
    """Aggregates frame-level features to single utterance-level feature.

    Reference:
    X-Vectors: Robust DNN Embeddings for Speaker Recognition
    https://www.danielpovey.com/files/2018_icassp_xvectors.pdf

    Args:
        input_size: Dimension of the input frame-level embeddings.
    """

    def __init__(self, input_size: int = 1536):
        super().__init__()
        self._output_size = input_size * 2

    def output_size(self):
        return self._output_size

    def forward(
        self,
        x: torch.Tensor,
        feat_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Forward pass of statistics pooling.

        Args:
            x: Input feature tensor of shape (batch_size, feature_dim, seq_len)
            feat_lengths: Optional tensor of shape (batch_size,) containing
                          the valid length of each sequence before padding

        Returns:
            x: Utterance-level embeddings of shape (batch_size, 2 × feature_dim)
        """
        if feat_lengths is not None:
            feat_lengths = feat_lengths.to(x.device)
            # Pooling over unpadded frames
            max_len = x.size(-1)
            mask = torch.arange(max_len, device=x.device).expand(
                x.size(0), max_len
            ) < feat_lengths.unsqueeze(1)
            mask = mask.unsqueeze(1)  # (B, 1, T)

            # Calculate mean over the time dimension (dim=-1)
            feat_lengths = feat_lengths.clamp(min=1)  # avoid division by zero
            mu = (x * mask).sum(dim=-1) / feat_lengths.unsqueeze(1)

            # Calculate standard deviation over the time dimension (dim=-1)
            # unbiased=False (divided by N rather than N - 1)
            variance = ((x - mu.unsqueeze(-1)) ** 2 * mask).sum(
                dim=-1
            ) / feat_lengths.unsqueeze(1)
            st = torch.sqrt(
                variance.clamp(min=torch.finfo(variance.dtype).eps, max=1e4)
            )  # add max clamp to prevent gradient explosion
        else:
            mu = torch.mean(x, dim=-1)
            variance = torch.var(x, dim=-1, unbiased=False)
            # add max clamp to prevent gradient explosion
            st = torch.sqrt(
                variance.clamp(min=torch.finfo(variance.dtype).eps, max=1e4)
            )

        x = torch.cat((mu, st), dim=1)

        return x
