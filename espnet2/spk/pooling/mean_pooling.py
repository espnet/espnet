import torch

from espnet2.spk.pooling.abs_pooling import AbsPooling


class MeanPooling(AbsPooling):
    """Average frame-level features to a single utterance-level feature.

    args:
        input_size: dimensionality of the input frame-level embeddings.
            Determined by encoder hyperparameter.
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
        self, x, task_tokens: torch.Tensor = None, mask: torch.Tensor = None, **kwargs
    ):
        """Forward.

        Args:
            x (torch.Tensor): Input tensor (#batch, size, time).
            task_tokens (torch.Tensor): Task tokens (#batch, size).
            mask (torch.Tensor): Mask tensor (#batch, time).
            **kwargs: additional keyword arguments (currently unused)
        """
        if task_tokens is not None:
            raise ValueError("MeanPooling is not adequate for task_tokens")
        if self.use_masking and mask is not None:
            x = x.masked_fill(mask.unsqueeze(1), 0)
            x = torch.sum(x, dim=-1)
            x = x / (torch.sum(~mask, dim=-1, keepdim=True) + 1e-6)
        else:
            x = torch.mean(x, dim=-1)

        return x
