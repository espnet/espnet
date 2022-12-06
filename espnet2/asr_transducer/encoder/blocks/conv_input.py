"""ConvInput block for Transducer encoder."""

from typing import Optional, Tuple, Union

import torch


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

        self.subsampling_factor = subsampling_factor
        self.vgg_like = vgg_like

        if vgg_like:
            conv_size1, conv_size2 = conv_size

            self.maxpool_kernel1, output_proj = self.get_module_parameters(
                input_size,
                conv_size2,
            )

            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, conv_size1, 3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(conv_size1, conv_size1, 3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(
                    self.maxpool_kernel1, stride=2, padding=0, ceil_mode=True
                ),
                torch.nn.Conv2d(conv_size1, conv_size2, 3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(conv_size2, conv_size2, 3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True),
            )
        else:
            (
                self.conv_kernel2,
                self.conv_stride2,
            ), output_proj = self.get_module_parameters(
                input_size,
                conv_size,
            )

            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, conv_size, 3, 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    conv_size, conv_size, self.conv_kernel2, self.conv_stride2
                ),
                torch.nn.ReLU(),
            )

        self.min_frame_length = 7 if subsampling_factor < 6 else 11

        if output_size is not None:
            self.output = torch.nn.Linear(output_proj, output_size)
            self.output_size = output_size
        else:
            self.output = None
            self.output_size = output_proj

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
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
            mask = mask[:, : x.size(1)]

        return x, mask

    def get_module_parameters(
        self,
        input_size: int,
        last_conv_size: int,
    ) -> Union[Tuple[int, int], Tuple[Tuple[int, int], int]]:
        """Return the convolution module parameters.

        Args:
            input_size: Module input size.
            last_conv_size: Last convolution size for module output size computation.

        Returns:
            : First MaxPool2D kernel size or second Conv2d kernel size and stride.
            output_size: Convolution module output size.

        """
        if self.vgg_like:
            maxpool_kernel1 = int(self.subsampling_factor / 2)

            output_size = last_conv_size * (((input_size - 1) // 2 - 1) // 2)

            return maxpool_kernel1, output_size

        if self.subsampling_factor == 2:
            conv_params = (3, 1)
        elif self.subsampling_factor == 4:
            conv_params = (3, 2)
        else:
            conv_params = (5, 3)

        output_size = last_conv_size * (
            ((input_size - 1) // 2 - (conv_params[0] - conv_params[1]))
            // conv_params[1]
        )

        return conv_params, output_size
