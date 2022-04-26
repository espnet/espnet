"""VGG2L block for Transducer encoder."""

from typing import Optional
from typing import Tuple
from typing import Union

import torch


class VGG2L(torch.nn.Module):
    """VGG2L module definition.

    Args:
        dim_input: Input dimension.
        mask_type: Type of mask for forward computation.
        subsampling_factor: Subsampling factor.
        pos_enc: Positional encoding class.
        dim_output: Block output dimension.

    """

    def __init__(
        self,
        dim_input: int,
        mask_type: str,
        subsampling_factor: int = 4,
        pos_enc: torch.nn.Module = None,
        dim_output: Optional[int] = None,
    ):
        """Construct a VGG2L object."""
        super().__init__()

        self.vgg2l = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 2)),
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
        )

        if pos_enc is not None:
            if dim_output is None:
                self.output = pos_enc
            else:
                self.output = torch.nn.Sequential(
                    torch.nn.Linear(128 * ((dim_input // 2) // 2), dim_output), pos_enc
                )
        else:
            self.output = None

        self.subsampling_factor = 4

        if mask_type == "rnn":
            self.create_new_mask = self.create_new_rnn_mask
        else:
            self.create_new_mask = self.create_new_conformer_mask

    def forward(
        self, sequence: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Encode input sequences.

        Args:
            sequence: VGG2L input sequences. (B, T, D_feats)
            mask: Mask of input sequences. (B, 1, T) or (B,)

        Returns:
            sequence: VGG2L output tensor.
                      (B, sub(T), D_out) or
                      ((B, sub(T), D_out), (B, 2 * (sub(T) - 1), D_pos))
            mask: Mask of output sequences. (B, 1, sub(T)) or (B,)

        """
        sequence = self.vgg2l(sequence.unsqueeze(1))

        b, c, t, f = sequence.size()

        sequence = sequence.transpose(1, 2).contiguous().view(b, t, c * f)

        if self.output is not None:
            sequence = self.output(sequence)

        return sequence, self.create_new_mask(mask)

    def create_new_conformer_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create a new conformer mask for output sequences.

        Args:
            mask: Mask of input sequences. (B, 1, T)

        Returns:
            mask: Mask of output sequences. (B, 1, sub(T))

        """
        vgg1_t_len = mask.size(2) - (mask.size(2) % 3)
        mask = mask[:, :, :vgg1_t_len][:, :, ::3]

        vgg2_t_len = mask.size(2) - (mask.size(2) % 2)
        mask = mask[:, :, :vgg2_t_len][:, :, ::2]

        return mask

    def create_new_rnn_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create new RNN mask for output sequences.

        Args:
            mask: Mask of input sequences. (B,)

        Returns:
            mask: Mask of output sequences. (B,)

        """
        mask = mask - (max(mask) % 3)
        mask = -(mask // -3)

        mask = mask - (max(mask) % 2)
        mask = -(mask // -2)

        return mask
