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
        hidden_size: dimensionality of the hidden layer
        use_masking: whether to use masking
    """

    def __init__(
        self, input_size: int = 1536, hidden_size: int = 128, use_masking: bool = False
    ):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_size * 3, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, input_size, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=2)
        self._output_size = input_size * 2
        self.use_masking = use_masking

    def output_size(self):
        return self._output_size

def forward(
    self,
    x: torch.Tensor,
    task_tokens: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    feat_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward pass of channel-attentive statistical pooling.

    Args:
        x (torch.Tensor): Input tensor (#batch, feature_dim, time).
        task_tokens (Optional[torch.Tensor]): Task tokens (#batch, size).
        mask (Optional[torch.Tensor]): Mask tensor (#batch, time).
        feat_lengths (Optional[torch.Tensor]): Valid length of each sequence (#batch,).

    Returns:
        torch.Tensor: Utterance-level embeddings (#batch, 2 × feature_dim).
    """
    if task_tokens is not None:
        raise ValueError(
            "ChannelAttentiveStatisticsPooling is not adequate for task_tokens"
        )

    T = x.size()[-1]
    
    # Handle masking based on available inputs
    if feat_lengths is not None:
        # Use feat_lengths approach (for spk version)
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
        
        # Create padding mask from feat_lengths
        padding_mask = torch.arange(T).expand(x.size(0), T).to(
            x.device
        ) >= feat_lengths.unsqueeze(1)
        
    elif self.use_masking and mask is not None:
        # Use mask approach (for uni_versa.sh)
        x_masked = x.masked_fill(mask.unsqueeze(1), 0)
        sum_x = torch.sum(x_masked, dim=-1)
        mean_x = sum_x / (torch.sum(~mask, dim=1, keepdim=True) + 1e-6)
        sum_var_x = torch.sum(
            torch.pow(x_masked - mean_x.unsqueeze(-1), 2).clamp(min=1e-4, max=1e4), 
            dim=-1
        )
        var_x = sum_var_x / (torch.sum(~mask, dim=1, keepdim=True) + 1e-6)
        std_x = torch.sqrt(var_x)
        global_x = torch.cat(
            (
                x,
                mean_x.unsqueeze(-1).repeat(1, 1, T),
                std_x.unsqueeze(-1).repeat(1, 1, T),
            ),
            dim=1,
        )
        padding_mask = mask
        
    else:
        # No masking
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
        padding_mask = None

    # Compute attention weights
    w = self.attention(global_x)
    
    # Apply masking to attention weights if available
    if padding_mask is not None:
        if padding_mask.dim() == 2:
            w = w.masked_fill(padding_mask.unsqueeze(1), torch.finfo(w.dtype).min)
        else:
            w = w.masked_fill(padding_mask, torch.finfo(w.dtype).min)
            
    w = self.softmax(w)

    # Compute weighted statistics
    mu = torch.sum(x * w, dim=2)
    sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4))
    x = torch.cat((mu, sg), dim=1)

    return x