import torch

from espnet2.layers.label_aggregation import LabelAggregate


class LabelProcessor(torch.nn.Module):
    """Label aggregator for speaker diarization"""

    def __init__(
        self, win_length: int = 512, hop_length: int = 128, center: bool = True
    ):
        super().__init__()
        self.label_aggregator = LabelAggregate(win_length, hop_length, center)

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input: (Batch, Nsamples, Label_dim)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Label_dim)
            olens: (Batch)

        """

        output, olens = self.label_aggregator(input, ilens)

        return output, olens
