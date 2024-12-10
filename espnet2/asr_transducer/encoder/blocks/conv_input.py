"""ConvInput block for Transducer encoder."""

from typing import Optional, Tuple, Union

import torch

from espnet2.asr_transducer.utils import get_convinput_module_parameters


class ConvInput(torch.nn.Module):
    """
    ConvInput block for Transducer encoder.

    This module defines a convolutional input block used in the Transducer encoder.
    It processes input sequences through a series of convolutional and pooling layers,
    optionally following a VGG-like architecture.

    Attributes:
        subsampling_factor (int): The factor by which the input sequence length is reduced.
        vgg_like (bool): Indicates if a VGG-like architecture is used.
        min_frame_length (int): The minimum frame length based on the subsampling factor.
        output_size (Optional[int]): The size of the output dimension after processing.

    Args:
        input_size (int): Size of the input feature dimension.
        conv_size (Union[int, Tuple[int]]): Size of the convolutional layers.
        subsampling_factor (int, optional): Factor for subsampling (default is 4).
        vgg_like (bool, optional): Flag to use VGG-like architecture (default is True).
        output_size (Optional[int], optional): Output dimension size (default is None).

    Examples:
        >>> conv_input = ConvInput(input_size=80, conv_size=(64, 128))
        >>> input_tensor = torch.randn(32, 100, 80)  # (B, T, D_feats)
        >>> output, mask = conv_input(input_tensor)

    Raises:
        ValueError: If `conv_size` does not match the expected format based on 
                    whether `vgg_like` is True or False.

    Note:
        The architecture is designed to handle both VGG-like structures and 
        standard convolutional structures based on the input parameters.
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

            self.maxpool_kernel1, output_proj = get_convinput_module_parameters(
                input_size, conv_size2, subsampling_factor, is_vgg=True
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
            ), output_proj = get_convinput_module_parameters(
                input_size, conv_size, subsampling_factor, is_vgg=False
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
        """
        ConvInput block for Transducer encoder.

        This module defines a convolutional input layer for the Transducer encoder. It 
        processes input sequences through convolutional layers and may apply subsampling 
        and pooling operations depending on the specified configuration.

        Attributes:
            subsampling_factor (int): The factor by which the input is subsampled.
            vgg_like (bool): Indicates whether a VGG-like architecture is used.
            output_size (Optional[int]): The output dimension of the block. If None, 
                it will be determined based on the convolutional output.

        Args:
            input_size (int): Size of the input feature vector.
            conv_size (Union[int, Tuple]): Size of the convolutional layers. If using 
                a VGG-like network, should be a tuple specifying sizes for two 
                convolutional layers.
            subsampling_factor (int, optional): Factor by which to subsample the input. 
                Default is 4.
            vgg_like (bool, optional): Whether to use a VGG-like architecture. Default 
                is True.
            output_size (Optional[int], optional): The desired output dimension. If 
                None, it is inferred from the convolutional layers.

        Examples:
            >>> conv_input = ConvInput(input_size=128, conv_size=(64, 128))
            >>> x = torch.randn(32, 10, 128)  # (Batch size, Time steps, Features)
            >>> mask = torch.ones(32, 1, 10)   # (Batch size, 1, Time steps)
            >>> output, output_mask = conv_input(x, mask)
            >>> print(output.shape)  # Should reflect the shape after conv layers and subsampling

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - x (torch.Tensor): Output sequences after convolution. Shape is 
                (B, sub(T), D_out).
                - mask (Optional[torch.Tensor]): Mask of the output sequences. Shape is 
                (B, 1, sub(T)) if mask is provided, otherwise None.

        Raises:
            ValueError: If the input tensor does not match the expected dimensions.
        """
        x = self.conv(x.unsqueeze(1))

        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)

        if self.output is not None:
            x = self.output(x)

        if mask is not None:
            mask = mask[:, : x.size(1)]

        return x, mask
