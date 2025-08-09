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
            # Pooling over unpadded frames
            mu = torch.stack(
                [
                    torch.mean(x[i, :, : int(l.item())], dim=-1)
                    for i, l in enumerate(feat_lengths)
                ],
                dim=0,
            )
            st = torch.stack(
                [
                    torch.std(x[i, :, : int(l.item())], dim=-1, unbiased=False)
                    for i, l in enumerate(feat_lengths)
                ],
                dim=0,
            )  # unbiased=False (divided by N rather than N - 1)
        else:
            mu = torch.mean(x, dim=-1)
            st = torch.std(x, dim=-1, unbiased=False)

        x = torch.cat((mu, st), dim=1)

        return x
