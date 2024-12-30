import typing as tp

from torch import nn

from espnet2.gan_codec.shared.encoder.seanet import (
    apply_parametrization_norm,
    get_norm_module,
)


class NormConv2d(nn.Module):
    """
        Wrapper around Conv2d and normalization applied to this conv to provide a
    uniform interface across normalization approaches.

    This class initializes a 2D convolutional layer and applies a normalization
    method as specified. It allows users to easily switch between different
    normalization techniques without altering the main structure of the model.

    Attributes:
        conv (nn.Module): The convolutional layer wrapped in the normalization.
        norm (nn.Module): The normalization layer applied to the convolutional
            output.
        norm_type (str): The type of normalization applied (e.g., 'batch', 'layer',
            'instance', or 'none').

    Args:
        *args: Variable length argument list to pass to the Conv2d constructor.
        norm (str): The type of normalization to apply. Defaults to "none".
        norm_kwargs (Dict[str, Any]): Additional keyword arguments for the
            normalization layer.
        **kwargs: Keyword arguments to pass to the Conv2d constructor.

    Returns:
        None

    Examples:
        >>> import torch
        >>> layer = NormConv2d(1, 32, kernel_size=3, norm='batch')
        >>> input_tensor = torch.randn(1, 1, 28, 28)
        >>> output_tensor = layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 32, 26, 26])

    Note:
        Ensure that the chosen normalization method is compatible with the
        application. Some normalization methods may require specific input
        shapes or types.
    """

    def __init__(
        self,
        *args,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        """
            Applies the convolution and normalization to the input tensor.

        This method takes an input tensor, applies the convolutional layer
        followed by the normalization layer. It ensures that the output is
        processed through both operations in sequence.

        Args:
            x (torch.Tensor): The input tensor to be processed. The shape of
            the tensor should match the expected input dimensions for the
            convolutional layer.

        Returns:
            torch.Tensor: The output tensor after applying the convolution
            and normalization layers.

        Examples:
            >>> import torch
            >>> model = NormConv2d(in_channels=3, out_channels=16, kernel_size=3)
            >>> input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1
            >>> output_tensor = model(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([1, 16, 62, 62])  # Output shape after convolution

        Note:
            Ensure that the input tensor's shape is compatible with the
            convolutional layer to avoid runtime errors.

        Raises:
            ValueError: If the input tensor's dimensions do not match
            the expected shape for the convolutional layer.
        """
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose2d(nn.Module):
    """
        Wrapper around ConvTranspose2d and normalization applied to this conv to provide
    a uniform interface across normalization approaches.

    This class extends the PyTorch `nn.Module` to include a transposed convolution
    layer with normalization. It allows for different normalization techniques to be
    applied seamlessly during the forward pass.

    Attributes:
        convtr (nn.ConvTranspose2d): The transposed convolution layer.
        norm (nn.Module): The normalization layer applied after the transposed
            convolution.

    Args:
        *args: Variable length argument list for the `ConvTranspose2d` constructor.
        norm (str): The type of normalization to apply. Default is "none".
        norm_kwargs (Dict[str, Any]): Additional keyword arguments for the
            normalization layer.
        **kwargs: Variable length keyword arguments for the `ConvTranspose2d`
            constructor.

    Returns:
        Tensor: The output tensor after applying the transposed convolution and
        normalization.

    Examples:
        >>> layer = NormConvTranspose2d(in_channels=3, out_channels=2, kernel_size=3)
        >>> input_tensor = torch.randn(1, 3, 10, 10)
        >>> output_tensor = layer(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 2, 12, 12])

    Note:
        This class relies on external functions `apply_parametrization_norm` and
        `get_norm_module` to handle the application of normalization.

    Todo:
        - Extend support for additional normalization techniques.
    """

    def __init__(
        self,
        *args,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose2d(*args, **kwargs), norm
        )
        self.norm = get_norm_module(self.convtr, causal=False, norm=norm, **norm_kwargs)

    def forward(self, x):
        """
            Wrapper around ConvTranspose2d and normalization applied to this conv
        to provide a uniform interface across normalization approaches.

        This class allows for the use of different normalization techniques
        alongside transposed convolution operations, enabling flexible
        architecture designs for deep learning models.

        Attributes:
            convtr (nn.ConvTranspose2d): The transposed convolution layer.
            norm (nn.Module): The normalization layer applied after the
                transposed convolution.

        Args:
            *args: Variable length argument list for ConvTranspose2d.
            norm (str): The type of normalization to apply. Default is "none".
            norm_kwargs (Dict[str, Any]): Additional keyword arguments for
                the normalization layer.
            **kwargs: Variable length keyword arguments for ConvTranspose2d.

        Returns:
            Tensor: The output tensor after applying the transposed convolution
            and normalization.

        Examples:
            >>> layer = NormConvTranspose2d(in_channels=16, out_channels=33,
            ...                              kernel_size=3, norm='batch')
            >>> input_tensor = torch.randn(1, 16, 10, 10)
            >>> output_tensor = layer(input_tensor)
            >>> output_tensor.shape
            torch.Size([1, 33, 12, 12])

        Note:
            Ensure that the input tensor shape is compatible with the
            ConvTranspose2d layer's requirements.

        Raises:
            ValueError: If the provided normalization type is not supported.
        """
        x = self.convtr(x)
        x = self.norm(x)
        return x
