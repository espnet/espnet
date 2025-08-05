from typing import Optional

import torch

from espnet2.spk.pooling.abs_pooling import AbsPooling


class MeanPooling(AbsPooling):
    """Average frame-level features to a single utterance-level feature.

    Args:
        input_size: Dimension of the input frame-level embeddings.
    """

    def __init__(self, input_size: int = 1536):
        super().__init__()
        self._output_size = input_size

    def output_size(self):
        return self._output_size

    def forward(
        self,
        x: torch.Tensor,
        feat_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of mean pooling.

        Args:
            x: Input feature tensor of shape (batch_size, feature_dim, seq_len)
            feat_lengths: Optional tensor of shape (batch_size,) containing
                          the valid length of each sequence before padding

        Returns:
            x: Utterance-level embeddings of shape (batch_size, feature_dim)
        """
        if feat_lengths is not None:
            # Pooling over unpadded frames
            x = torch.stack(
                [
                    torch.mean(x[i, :, : int(l.item())], dim=-1)
                    for i, l in enumerate(feat_lengths)
                ],
                dim=0,
            )
        else:
            x = torch.mean(x, dim=-1)

        return x
