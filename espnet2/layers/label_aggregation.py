from typing import Optional, Tuple

import torch
from typeguard import typechecked

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class LabelAggregate(torch.nn.Module):
    """
        LabelAggregate is a PyTorch module that performs label aggregation over input
    sequences. It processes the input tensor to produce aggregated labels based on
    the specified window and hop lengths, facilitating tasks such as speech
    recognition and other sequence labeling applications.

    Attributes:
        win_length (int): The length of the window used for aggregation.
        hop_length (int): The hop length used to slide the window across the input.
        center (bool): If True, pads the input tensor on both sides before
            aggregation.

    Args:
        win_length (int, optional): The length of the window for aggregation.
            Default is 512.
        hop_length (int, optional): The hop length for sliding the window.
            Default is 128.
        center (bool, optional): Whether to pad the input tensor symmetrically.
            Default is True.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - output (torch.Tensor): The aggregated label tensor of shape
              (Batch, Frames, Label_dim).
            - olens (Optional[torch.Tensor]): The lengths of the output sequences
              if ilens is provided, otherwise None.

    Examples:
        >>> label_aggregate = LabelAggregate(win_length=256, hop_length=64)
        >>> input_tensor = torch.rand(10, 1000, 20)  # (Batch, Nsamples, Label_dim)
        >>> ilens = torch.tensor([1000] * 10)  # Lengths for each sequence in the batch
        >>> output, olens = label_aggregate(input_tensor, ilens)
        >>> print(output.shape)  # Output shape should be (10, Frames, 20)

    Note:
        The default behavior of label aggregation is compatible with
        torch.stft regarding framing and padding.
    """

    @typechecked
    def __init__(
        self,
        win_length: int = 512,
        hop_length: int = 128,
        center: bool = True,
    ):
        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center

    def extra_repr(self):
        """
                Returns a string representation of the LabelAggregate parameters.

        This method provides a formatted string that includes the values of the
        `win_length`, `hop_length`, and `center` attributes of the LabelAggregate
        instance, which can be useful for debugging and logging purposes.

        Attributes:
            win_length (int): The length of the window used for aggregation.
            hop_length (int): The number of samples to hop for each frame.
            center (bool): Whether to pad the input tensor on both sides.

        Returns:
            str: A string representation of the LabelAggregate parameters.

        Examples:
            >>> label_agg = LabelAggregate(win_length=512, hop_length=128, center=True)
            >>> print(label_agg.extra_repr())
            win_length=512, hop_length=128, center=True
        """
        return (
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
                LabelAggregate forward function.

        This method processes the input tensor through a series of steps to perform
        label aggregation, which is useful in tasks such as speech processing. The
        forward function takes an input tensor and an optional lengths tensor,
        returns the aggregated output and the processed lengths.

        Args:
            input: A tensor of shape (Batch, Nsamples, Label_dim) representing the
                input data.
            ilens: An optional tensor of shape (Batch) that represents the lengths
                of each input sequence.

        Returns:
            output: A tensor of shape (Batch, Frames, Label_dim) containing the
                aggregated labels.
            Optional[torch.Tensor]: A tensor of processed lengths, shape (Batch)
                if `ilens` is provided, otherwise None.

        Examples:
            >>> label_aggregate = LabelAggregate(win_length=512, hop_length=128)
            >>> input_tensor = torch.randn(2, 1000, 10)  # Batch size 2, 1000 samples, 10 labels
            >>> ilens = torch.tensor([1000, 800])  # Lengths of each sequence
            >>> output, olens = label_aggregate(input_tensor, ilens)
            >>> print(output.shape)  # Should print: torch.Size([2, Frames, 10])
            >>> print(olens)  # Lengths of the output sequences if ilens is provided
        """
        bs = input.size(0)
        max_length = input.size(1)
        label_dim = input.size(2)

        # NOTE(jiatong):
        #   The default behaviour of label aggregation is compatible with
        #   torch.stft about framing and padding.

        # Step1: center padding
        if self.center:
            pad = self.win_length // 2
            max_length = max_length + 2 * pad
            input = torch.nn.functional.pad(input, (0, 0, pad, pad), "constant", 0)
            input[:, :pad, :] = input[:, pad : (2 * pad), :]
            input[:, (max_length - pad) : max_length, :] = input[
                :, (max_length - 2 * pad) : (max_length - pad), :
            ]
            nframe = (
                torch.div(
                    max_length - self.win_length, self.hop_length, rounding_mode="trunc"
                )
                + 1
            )

        # Step2: framing
        output = input.as_strided(
            (bs, nframe, self.win_length, label_dim),
            (max_length * label_dim, self.hop_length * label_dim, label_dim, 1),
        )

        # Step3: aggregate label
        output = torch.gt(output.sum(dim=2, keepdim=False), self.win_length // 2)
        output = output.float()

        # Step4: process lengths
        if ilens is not None:
            if self.center:
                pad = self.win_length // 2
                ilens = ilens + 2 * pad

            olens = (
                torch.div(
                    ilens - self.win_length, self.hop_length, rounding_mode="trunc"
                )
                + 1
            )
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens
