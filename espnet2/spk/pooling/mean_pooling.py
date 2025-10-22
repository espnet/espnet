from typing import Optional

import torch

from espnet2.spk.pooling.abs_pooling import AbsPooling


class MeanPooling(AbsPooling):
    """Average frame-level features to a single utterance-level feature.

    Args:
        input_size: Dimension of the input frame-level embeddings.
        use_masking: whether to use masking for pooling.
        **kwargs: additional keyword arguments (currently unused but accepted
            for compatibility)
    """

    def __init__(self, input_size: int = 1536, use_masking: bool = False, **kwargs):
        super().__init__()
        self._output_size = input_size
        self.use_masking = use_masking

    def output_size(self):
        return self._output_size

    def forward(
        self,
        x: torch.Tensor,
        task_tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        feat_lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for mean pooling.

        Args:
            x (torch.Tensor): Input tensor of shape (#batch, feature_dim, seq_len).
            task_tokens (Optional[torch.Tensor]): Task tokens (#batch, feature_dim).
            mask (Optional[torch.Tensor]): Boolean mask tensor (#batch, seq_len)
                                           where True indicates padded positions.
            feat_lengths (Optional[torch.Tensor]): Tensor of shape (#batch,) containing
                                                  the valid length of each sequence.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            torch.Tensor: Utterance-level embeddings of shape (#batch, feature_dim).

        Raises:
            ValueError: If task_tokens are provided (not supported by mean pooling).
        """
        if task_tokens is not None:
            raise ValueError("MeanPooling is not adequate for task_tokens")

        # Option 1: Use feat_lengths if provided (most efficient for variable length sequences)
        if feat_lengths is not None:
            # Pooling over unpadded frames using actual sequence lengths
            x = torch.stack(
                [
                    torch.mean(x[i, :, : int(l.item())], dim=-1)
                    for i, l in enumerate(feat_lengths)
                ],
                dim=0,
            )
        # Option 2: Use mask if provided and masking is enabled
        elif self.use_masking and mask is not None:
            # Apply mask to zero out padded positions
            x = x.masked_fill(mask.unsqueeze(1), 0)
            # Sum over sequence dimension
            x = torch.sum(x, dim=-1)
            # Divide by the number of valid (non-masked) positions
            valid_lengths = torch.sum(~mask, dim=-1, keepdim=True) + 1e-6
            x = x / valid_lengths
        # Option 3: Simple mean over all positions (when no masking info is available)
        else:
            x = torch.mean(x, dim=-1)

        return x
