"""ConvInput block for Transducer encoder."""

from typing import Optional, Tuple, Union

import torch

from espnet2.asr_transducer.utils import sub_factor_to_params


class ConvInput(torch.nn.Module):
    """ConvInput module definition.

    Args:
        input_size: Input size.
        conv_size: Convolution size.
        subsampling_factor: Subsampling factor.
        vgg_like: Whether to use a VGG-like network.
        output_size: Block output dimension.

    """

    def __init__(
        self,
        input_size: int,
        conv_size: Union[int, Tuple],
        subsampling_factor: int = 4,
        vgg_like: bool = True,
        output_size: Optional[int] = None,
    ) -> None:
        """Construct a ConvInput object."""
        super().__init__()

        if vgg_like:
            conv_size1, conv_size2 = conv_size

            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, conv_size1, 3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(conv_size1, conv_size1, 3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((3, 2)),
                torch.nn.Conv2d(conv_size1, conv_size2, 3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(conv_size2, conv_size2, 3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),
            )

            output_proj = conv_size2 * ((input_size // 2) // 2)

            self.subsampling_factor = 4

            self.create_new_mask = self.create_new_vgg_mask
        else:
            kernel_2, stride_2, conv_2_output_size = sub_factor_to_params(
                subsampling_factor,
                input_size,
            )

            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, conv_size, 3, 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(conv_size, conv_size, kernel_2, stride_2),
                torch.nn.ReLU(),
            )

            output_proj = conv_size * conv_2_output_size

            self.subsampling_factor = subsampling_factor
            self.kernel_2 = kernel_2
            self.stride_2 = stride_2

            self.create_new_mask = self.create_new_conv2d_mask

        self.vgg_like = vgg_like
        self.min_frame_length = 7 if subsampling_factor < 6 else 11

        if output_size is not None:
            self.output = torch.nn.Linear(output_proj, output_size)
            self.output_size = output_size
        else:
            self.output = None
            self.output_size = output_proj

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequences.

        Args:
            x: ConvInput input sequences. (B, T, D_feats)
            mask: Mask of input sequences. (B, 1, T)

        Returns:
            x: ConvInput output sequences. (B, sub(T), D_out)
            mask: Mask of output sequences. (B, 1, sub(T))

        """
        x = self.conv(x.unsqueeze(1))

        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)

        if self.output is not None:
            x = self.output(x)

        if mask is not None:
            mask = self.create_new_mask(mask)

        return x, mask

    def create_new_vgg_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create a new mask for VGG output sequences.

        Args:
            mask: Mask of input sequences. (B, T)

        Returns:
            mask: Mask of output sequences. (B, sub(T))

        """
        vgg1_t_len = mask.size(1) - (mask.size(1) % 3)
        mask = mask[:, :vgg1_t_len][:, ::3]

        vgg2_t_len = mask.size(1) - (mask.size(1) % 2)
        mask = mask[:, :vgg2_t_len][:, ::2]

        return mask

    def create_new_conv2d_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create new conformer mask for Conv2d output sequences.

        Args:
            mask: Mask of input sequences. (B, T)

        Returns:
            mask: Mask of output sequences. (B, sub(T))

        """
        return mask[:, :-2:2][:, : -(self.kernel_2 - 1) : self.stride_2]

    def get_size_before_subsampling(self, size: int) -> int:
        """Return the original size before subsampling for a given size.

        Args:
            size: Number of frames after subsampling.

        Returns:
            : Number of frames before subsampling.

        """
        if self.vgg_like:
            return ((size * 2) * 3) + 1

        return ((size + 2) * 2) + (self.kernel_2 - 1) * self.stride_2
