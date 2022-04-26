"""Conv2DSubsampling block for Transducer encoder."""

from typing import Optional
from typing import Tuple
from typing import Union

import torch

from espnet2.asr_transducer.utils import sub_factor_to_params


class Conv2dSubsampling(torch.nn.Module):
    """Conv2DSubsampling module definition.

    Args:
        dim_input: Input dimension.
        mask_type: Type of mask for forward computation.
        dim_conv: Convolution output dimension.
        subsampling_factor: Subsampling factor for convolution.
        pos_enc: Positional encoding class.
        dim_output: Block output dimension.

    """

    def __init__(
        self,
        dim_input: int,
        mask_type: str,
        dim_conv: int = 256,
        subsampling_factor: int = 4,
        pos_enc: torch.nn.Module = None,
        dim_output: Optional[int] = None,
    ):
        """Construct an Conv2dSubsampling object."""
        super().__init__()

        kernel_2, stride_2, conv_2_dim_output = sub_factor_to_params(
            subsampling_factor,
            dim_input,
        )

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, dim_conv, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim_conv, dim_conv, kernel_2, stride_2),
            torch.nn.ReLU(),
        )

        if pos_enc is not None:
            if dim_output is None:
                self.output = pos_enc
            else:
                self.output = torch.nn.Sequential(
                    torch.nn.Linear((dim_conv * conv_2_dim_output), dim_output), pos_enc
                )
        else:
            self.output = None

        self.subsampling_factor = subsampling_factor

        self.kernel_2 = kernel_2
        self.stride_2 = stride_2

        if mask_type == "rnn":
            self.create_new_mask = self.create_new_rnn_mask
        else:
            self.create_new_mask = self.create_new_conformer_mask

    def forward(
        self, sequence: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Encode input sequences.

        Args:
            sequence: Conv2dSubsampling input sequences. (B, T, D_feats)
            mask: Mask of input sequences. (B, 1, F) or (B,)

        Returns:
            sequence: Conv2dSubsampling output tensor.
                      (B, sub(T), D_output) or
                      ((B, sub(T), D_out), (B, 2 * (sub(T) - 1), D_pos))
            mask: Mask of output sequences. (B, 1, sub(T)) or (B,)

        """
        sequence = sequence.unsqueeze(1)
        sequence = self.conv(sequence)

        b, c, t, f = sequence.size()

        sequence = sequence.transpose(1, 2).contiguous().view(b, t, c * f)

        if self.output is not None:
            sequence = self.output(sequence)

        return sequence, self.create_new_mask(mask)

    def create_new_conformer_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create new conformer mask for output sequences.

        Args:
            x_mask: Mask of input sequences. (B, 1, F)

        Returns:
            x_mask: Mask of output sequences. (B, 1, F // subsampling_factor)

        """
        return mask[:, :, :-2:2][:, :, : -(self.kernel_2 - 1) : self.stride_2]

    def create_new_rnn_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create new RNN mask for output sequences.

        Args:
            mask: Mask of input sequences. (B,)

        Returns:
            mask: Mask of output sequences. (B,)

        """
        mask = (mask - 1) // 2
        mask = (mask - (self.kernel_2 - 1)) // self.stride_2

        return mask
