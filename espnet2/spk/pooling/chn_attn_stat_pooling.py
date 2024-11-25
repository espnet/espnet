import torch
import torch.nn as nn

from espnet2.spk.pooling.abs_pooling import AbsPooling


class ChnAttnStatPooling(AbsPooling):
    """Aggregates frame-level features to single utterance-level feature.

    Proposed in B.Desplanques et al., "ECAPA-TDNN: Emphasized Channel
    Attention, Propagation and Aggregation in TDNN Based Speaker Verification"

    args:
        input_size: dimensionality of the input frame-level embeddings.
            Determined by encoder hyperparameter.
            For this pooling layer, the output dimensionality will be double of
            the input_size
        hidden_size: dimensionality of the hidden layer
        use_masking: whether to use masking
    """

    def __init__(self, input_size: int = 1536, hidden_size: int = 128, use_masking: bool = False):
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

    def forward(self, x, task_tokens: torch.Tensor = None, mask: torch.Tensor = None):
        """Forward.

        Args:
            x (torch.Tensor): Input tensor (#batch, size, time).
            task_tokens (torch.Tensor): Task tokens (#batch, size).
            mask (torch.Tensor): Mask tensor (#batch, time).
        """
        if task_tokens is not None:
            raise ValueError(
                "ChannelAttentiveStatisticsPooling is not adequate for task_tokens"
            )
        
        t = x.size()[-1]
        if self.use_masking and mask is not None:
            x = x.masked_fill(mask.unsqueeze(1), 0)
            sum_x = torch.sum(x, dim=-1)
            mean_x = sum_x / (torch.sum(~mask, dim=1, keepdim=True) + 1e-6)
            var_x = torch.power(x - mean_x, 2).clamp(min=1e-4, max=1e4)
            std_x = torch.sqrt(var_x)
            global_x = torch.cat(
                (
                    x,
                    mean_x.repeat(1, 1, t),
                    std_x.repeat(1, 1, t),
                ),
                dim=1,
            )
        else:
            global_x = torch.cat(
                (
                    x,
                    torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                    torch.sqrt(
                        torch.var(x, dim=2, keepdim=True).clamp(min=1e-4, max=1e4)
                    ).repeat(1, 1, t),
                ),
                dim=1,
            )

        w = self.attention(global_x)
        if self.use_masking and mask is not None:
            w = w.masked_fill(mask.unsqueeze(1), -1e9)
        w = self.softmax(w)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4))
        x = torch.cat((mu, sg), dim=1)

        return x
