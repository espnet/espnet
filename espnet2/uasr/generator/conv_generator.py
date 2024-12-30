import argparse
import logging
from typing import Dict, Optional

import torch
from typeguard import typechecked

from espnet2.uasr.generator.abs_generator import AbsGenerator
from espnet2.utils.types import str2bool


class TransposeLast(torch.nn.Module):
    """
    Transpose the last two dimensions of the input tensor.

    This module is designed to facilitate the manipulation of tensor shapes
    in neural network architectures. It can also deconstruct the input tensor
    by selecting a specific index if `deconstruct_idx` is provided.

    Attributes:
        deconstruct_idx (Optional[int]): The index to deconstruct the input
            tensor. If None, no deconstruction is performed.

    Args:
        deconstruct_idx (Optional[int]): The index to select a specific part
            of the input tensor. Defaults to None.

    Returns:
        torch.Tensor: The input tensor with its last two dimensions transposed.

    Examples:
        >>> import torch
        >>> transpose_last = TransposeLast()
        >>> x = torch.randn(2, 3, 4)
        >>> output = transpose_last(x)
        >>> output.shape
        torch.Size([2, 4, 3])

        >>> transpose_last_deconstruct = TransposeLast(deconstruct_idx=1)
        >>> x = torch.randn(2, 3, 4)
        >>> output = transpose_last_deconstruct(x)
        >>> output.shape
        torch.Size([2, 4])
    """

    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        """
            Forward pass for the convolutional generator.

        This method processes the input features through a series of operations,
        including optional batch normalization, residual connections, and
        convolutional layers, to generate the output samples. It also computes
        the real sample if text input is provided.

        Args:
            feats (torch.Tensor): Input feature tensor of shape (B, C, L), where
                B is the batch size, C is the number of channels, and L is the
                length of the input sequence.
            text (Optional[torch.Tensor]): Input tensor containing text indices.
                If provided, it is used to generate a real sample.
            feats_padding_mask (torch.Tensor): Padding mask tensor of shape (B, L)
                indicating which features are valid (True) or padded (False).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
            torch.Tensor]: A tuple containing:
                - generated_sample (torch.Tensor): Output tensor after processing,
                  shape (B, output_dim, new_length).
                - real_sample (Optional[torch.Tensor]): Tensor representing the
                  one-hot encoded real sample if text is provided, otherwise None.
                - inter_x (Optional[torch.Tensor]): Intermediate tensor used in
                  residual connection if applicable, otherwise None.
                - generated_sample_padding_mask (torch.Tensor): Updated padding
                  mask for the generated sample, shape (B, new_length).

        Raises:
            AssertionError: If the text tensor is provided but contains no non-zero
                elements.

        Examples:
            >>> generator = ConvGenerator(input_dim=256, output_dim=512)
            >>> feats = torch.randn(10, 256, 50)
            >>> text = torch.tensor([[1, 0], [0, 1]])
            >>> feats_padding_mask = torch.ones(10, 50, dtype=torch.bool)
            >>> generated_sample, real_sample, inter_x, mask = generator.forward(
            ...     feats, text, feats_padding_mask
            ... )

        Note:
            This function is designed to work with batch processing of inputs.
            Ensure that the input tensors have the correct dimensions to avoid
            runtime errors.
        """
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class SamePad(torch.nn.Module):
    """
    Applies same padding to the input tensor based on the kernel size.

    This module ensures that the output tensor has the same spatial dimensions
    as the input tensor after applying a convolution operation, by calculating
    the necessary padding based on the kernel size. It can also handle causal
    padding if specified.

    Attributes:
        remove (int): The number of elements to remove from the end of the input
            tensor based on the kernel size and whether causal padding is
            applied.

    Args:
        kernel_size (int): The size of the convolution kernel.
        causal (bool): If True, applies causal padding. Defaults to False.

    Returns:
        torch.Tensor: The padded input tensor with the same spatial dimensions.

    Examples:
        >>> import torch
        >>> same_pad = SamePad(kernel_size=3)
        >>> input_tensor = torch.randn(1, 3, 10)  # (batch_size, channels, length)
        >>> output_tensor = same_pad(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 3, 10])  # Output shape is the same as input shape

        >>> causal_pad = SamePad(kernel_size=3, causal=True)
        >>> output_tensor_causal = causal_pad(input_tensor)
        >>> output_tensor_causal.shape
        torch.Size([1, 3, 9])  # Output shape is reduced by 2 for causal padding
    """

    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        """
            Forward pass for the ConvGenerator.

        This method processes input features through a series of transformations,
        including optional batch normalization, residual connections, and convolution.
        It generates output samples along with a real sample if text is provided,
        and adjusts the padding mask accordingly.

        Args:
            feats (torch.Tensor): Input feature tensor of shape (batch_size,
                input_dim, seq_length).
            text (Optional[torch.Tensor]): Optional tensor of shape (batch_size,
                text_length) containing target text indices.
            feats_padding_mask (torch.Tensor): Tensor of shape (batch_size,
                seq_length) indicating padded positions (True for padding,
                False for valid data).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
                torch.Tensor]: A tuple containing:
                - generated_sample (torch.Tensor): The generated output tensor
                    of shape (batch_size, output_dim, new_seq_length).
                - real_sample (Optional[torch.Tensor]): The one-hot encoded tensor
                    of the target text if text is provided, otherwise None.
                - inter_x (Optional[torch.Tensor]): Intermediate output from the
                    residual connection if enabled, otherwise None.
                - generated_sample_padding_mask (torch.Tensor): The updated
                    padding mask for the generated sample.

        Raises:
            AssertionError: If `text` is provided but contains only zeros.

        Examples:
            >>> generator = ConvGenerator(input_dim=256, output_dim=128)
            >>> feats = torch.randn(32, 256, 100)
            >>> text = torch.randint(0, 128, (32, 10))
            >>> padding_mask = torch.ones(32, 100, dtype=torch.bool)
            >>> generated_sample, real_sample, inter_x, mask = generator(feats,
            ... text, padding_mask)

        Note:
            This method assumes that the input tensors are properly shaped and
            that the padding mask correctly reflects the padded regions of the
            input features.
        """
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class ConvGenerator(AbsGenerator):
    """
        ConvGenerator is a convolutional generator for Unsupervised Automatic Speech
    Recognition (UASR). It inherits from the AbsGenerator class and is designed to
    generate output features from input audio features through a series of
    convolutional operations.

    Attributes:
        input_dim (int): The dimension of the input features.
        output_dim (int): The dimension of the output features.
        conv_kernel (int): The kernel size for the convolutional layer.
        conv_dilation (int): The dilation rate for the convolutional layer.
        conv_stride (int): The stride for the convolutional layer.
        pad (int): The padding value for the convolutional layer.
        bias (bool): Whether to include a bias term in the convolutional layer.
        dropout (torch.nn.Dropout): Dropout layer for regularization.
        batch_norm (bool): Whether to use batch normalization.
        batch_norm_weight (float): The weight for batch normalization.
        residual (bool): Whether to use a residual connection.

    Args:
        input_dim (int): The dimension of the input features.
        output_dim (int): The dimension of the output features.
        cfg (Optional[Dict]): Configuration dictionary for initializing the generator.
        conv_kernel (int): Kernel size for the convolutional layer (default: 3).
        conv_dilation (int): Dilation rate for the convolutional layer (default: 1).
        conv_stride (int): Stride for the convolutional layer (default: 9).
        pad (int): Padding value for the convolutional layer (default: -1).
        bias (str2bool): Whether to include bias in convolution (default: False).
        dropout (float): Dropout rate (default: 0.0).
        batch_norm (str2bool): Whether to use batch normalization (default: True).
        batch_norm_weight (float): Weight for batch normalization (default: 30.0).
        residual (str2bool): Whether to use a residual connection (default: True).

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
        torch.Tensor]: A tuple containing the generated sample, real sample (if
        text is provided), intermediate features (if residual is used), and
        generated sample padding mask.

    Raises:
        AssertionError: If the text tensor contains no non-zero elements when
        generating the real sample.

    Examples:
        # Initialize the ConvGenerator
        generator = ConvGenerator(input_dim=256, output_dim=128)

        # Forward pass through the generator
        generated_sample, real_sample, inter_x, padding_mask = generator(
            feats=torch.randn(10, 256, 100),
            text=torch.tensor([[1, 0, 0], [0, 1, 0]]),
            feats_padding_mask=torch.ones(10, 100).bool()
        )

    Note:
        The generated sample and padding mask will be based on the convolutional
        operations applied to the input features. The real sample is constructed
        based on the provided text, and it is expected that the text tensor has
        non-zero elements.
    """

    @typechecked
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: Optional[Dict] = None,
        conv_kernel: int = 3,
        conv_dilation: int = 1,
        conv_stride: int = 9,
        pad: int = -1,
        bias: str2bool = False,
        dropout: float = 0.0,
        batch_norm: str2bool = True,
        batch_norm_weight: float = 30.0,
        residual: str2bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if cfg is not None:
            cfg = argparse.Namespace(**cfg)
            self.conv_kernel = cfg.generator_kernel
            self.conv_dilation = cfg.generator_dilation
            self.conv_stride = cfg.generator_stride
            self.pad = cfg.generator_pad
            self.bias = cfg.generator_bias
            self.dropout = torch.nn.Dropout(cfg.generator_dropout)
            # TODO(Dongji): batch_norm is not in cfg
            self.batch_norm = False
            self.batch_norm_weight = cfg.generator_batch_norm
            self.residual = cfg.generator_residual
        else:
            self.conv_kernel = conv_kernel
            self.conv_dilation = conv_dilation
            self.conv_stride = conv_stride
            self.output_dim = output_dim
            self.pad = pad
            self.bias = bias
            self.dropout = torch.nn.Dropout(dropout)
            self.batch_norm = batch_norm
            self.batch_norm_weight = batch_norm_weight
            self.residual = residual

        if self.pad < 0:
            self.padding = self.conv_kernel // 2
        else:
            self.padding = self.pad

        self.proj = torch.nn.Sequential(
            TransposeLast(),
            torch.nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                dilation=self.conv_dilation,
                padding=self.padding,
                bias=self.bias,
            ),
            TransposeLast(),
        )
        if self.batch_norm:
            self.bn = torch.nn.BatchNorm1d(input_dim)
            self.bn.weight.data.fill_(self.batch_norm_weight)
        if self.residual:
            self.in_proj = torch.nn.Linear(input_dim, input_dim)

    def output_size(self):
        """
            Returns the output dimension of the convolutional generator.

        This property is useful to retrieve the output size after the convolutional
        layers have been applied, particularly in scenarios where the generator is
        part of a larger model and the output dimensions need to be known for
        subsequent processing steps.

        Returns:
            int: The output dimension of the generator.

        Examples:
            generator = ConvGenerator(input_dim=128, output_dim=256)
            output_dim = generator.output_size()  # output_dim will be 256
        """
        return self.output_dim

    def forward(
        self,
        feats: torch.Tensor,
        text: Optional[torch.Tensor],
        feats_padding_mask: torch.Tensor,
    ):
        """
            Perform the forward pass of the convolutional generator.

        This method processes the input features and generates output samples
        using convolutional layers. It can optionally incorporate batch normalization
        and residual connections based on the initialization parameters.

        Args:
            feats (torch.Tensor): Input tensor of shape (batch_size, input_dim, seq_len).
            text (Optional[torch.Tensor]): Optional tensor of shape (batch_size, seq_len).
                Used to create a one-hot representation of the target outputs.
            feats_padding_mask (torch.Tensor): A boolean mask of shape (batch_size, seq_len)
                indicating which elements are valid (True) or padded (False).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
                  torch.Tensor]: A tuple containing:
                  - generated_sample (torch.Tensor): Output tensor of shape
                    (batch_size, output_dim, new_seq_len) with generated samples.
                  - real_sample (Optional[torch.Tensor]): One-hot encoded tensor of
                    shape (batch_size, seq_len, output_dim) for real samples.
                    Returns None if text is None.
                  - inter_x (Optional[torch.Tensor]): Intermediate tensor from the
                    residual connection. Returns None if residual is not used.
                  - generated_sample_padding_mask (torch.Tensor): A mask for the
                    generated samples of shape (batch_size, new_seq_len) indicating
                    valid elements.

        Raises:
            AssertionError: If the input text tensor contains no non-zero elements
            or if the size of generated_sample_padding_mask does not match the
            expected output shape.

        Examples:
            >>> generator = ConvGenerator(input_dim=256, output_dim=128)
            >>> feats = torch.randn(32, 256, 50)  # Example input features
            >>> text = torch.randint(0, 128, (32, 50))  # Example text input
            >>> feats_padding_mask = torch.ones(32, 50, dtype=torch.bool)  # No padding
            >>> output = generator.forward(feats, text, feats_padding_mask)
            >>> generated_sample, real_sample, inter_x, mask = output
        """
        inter_x = None
        if self.batch_norm:
            feats = self.bn_padded_data(feats, feats_padding_mask)
        if self.residual:
            inter_x = self.in_proj(self.dropout(feats))
            feats = feats + inter_x

        feats = self.dropout(feats)

        generated_sample = self.proj(feats)
        generated_sample_padding_mask = feats_padding_mask[:, :: self.conv_stride]

        if generated_sample_padding_mask.size(1) != generated_sample.size(1):
            new_padding = generated_sample_padding_mask.new_zeros(
                generated_sample.shape[:-1]
            )
            diff = new_padding.size(1) - generated_sample_padding_mask.size(1)

            if diff > 0:
                new_padding[:, diff:] = generated_sample_padding_mask
            else:
                logging.info("ATTENTION: make sure that you are using V2 instead of V1")
                assert diff < 0
                new_padding = generated_sample_padding_mask[:, :diff]

            generated_sample_padding_mask = new_padding

        real_sample = None
        if text is not None:
            assert torch.count_nonzero(text) > 0
            real_sample = generated_sample.new_zeros(text.numel(), self.output_dim)
            real_sample.scatter_(1, text.view(-1, 1).long(), 1)
            real_sample = real_sample.view(text.shape + (self.output_dim,))

        return generated_sample, real_sample, inter_x, generated_sample_padding_mask

    def bn_padded_data(self, feature: torch.Tensor, padding_mask: torch.Tensor):
        """
            Normalize the input features using batch normalization while considering
        the padding mask.

        This method applies batch normalization to the input feature tensor,
        only for the elements that are not masked by the padding mask. The
        elements corresponding to the padding mask are left unchanged.

        Args:
            feature (torch.Tensor): The input feature tensor of shape (B, C, L),
                where B is the batch size, C is the number of channels, and L is
                the length of the sequence.
            padding_mask (torch.Tensor): A boolean tensor of shape (B, L) that
                indicates which elements in the feature tensor should be considered
                for normalization. Elements with a value of `True` are included in
                the normalization, while those with `False` are ignored.

        Returns:
            torch.Tensor: The normalized feature tensor of the same shape as the
            input feature tensor, where the non-masked elements have been batch
            normalized.

        Examples:
            >>> import torch
            >>> bn_layer = ConvGenerator(input_dim=64, output_dim=32)
            >>> features = torch.randn(10, 64, 100)
            >>> padding_mask = torch.ones(10, 100, dtype=torch.bool)
            >>> padding_mask[:, 10:] = False
            >>> normalized_features = bn_layer.bn_padded_data(features, padding_mask)
            >>> print(normalized_features.shape)
            torch.Size([10, 64, 100])
        """
        normed_feature = feature.clone()
        normed_feature[~padding_mask] = self.bn(
            feature[~padding_mask].unsqueeze(-1)
        ).squeeze(-1)
        return normed_feature
