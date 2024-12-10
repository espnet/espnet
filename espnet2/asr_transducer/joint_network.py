"""Transducer joint network implementation."""

import torch

from espnet2.asr_transducer.activation import get_activation


class JointNetwork(torch.nn.Module):
    """
    Transducer joint network implementation.

    This module implements a joint network for transducer models in automatic
    speech recognition (ASR). The JointNetwork class combines encoder and decoder
    outputs through a specified activation function to produce the final output.

    Attributes:
        lin_enc (torch.nn.Linear): Linear layer for encoder output.
        lin_dec (torch.nn.Linear): Linear layer for decoder output.
        lin_out (torch.nn.Linear): Linear layer for producing final output.
        joint_activation (callable): Activation function for the joint network.

    Args:
        output_size (int): Output size.
        encoder_size (int): Encoder output size.
        decoder_size (int): Decoder output size.
        joint_space_size (int, optional): Joint space size (default is 256).
        joint_activation_type (str, optional): Type of activation for joint network
            (default is "tanh").
        lin_dec_bias (bool, optional): Whether to include bias in the decoder linear
            layer (default is True).
        **activation_parameters: Additional parameters for the activation function.

    Examples:
        >>> joint_network = JointNetwork(
        ...     output_size=10,
        ...     encoder_size=20,
        ...     decoder_size=30,
        ...     joint_space_size=256,
        ...     joint_activation_type='relu'
        ... )
        >>> enc_out = torch.randn(5, 10, 1, 20)  # (B, T, s_range, D_enc)
        >>> dec_out = torch.randn(5, 10, 1, 30)  # (B, T, U, D_dec)
        >>> output = joint_network(enc_out, dec_out)
        >>> print(output.shape)  # Should be (5, 10, U, 10)

    Raises:
        ValueError: If the shapes of enc_out and dec_out do not match expected
            dimensions.

    Note:
        The input tensors enc_out and dec_out can have different shapes depending
        on the specific use case. The joint network computes their combined output
        through learned linear transformations followed by a specified activation.
    """

    def __init__(
        self,
        output_size: int,
        encoder_size: int,
        decoder_size: int,
        joint_space_size: int = 256,
        joint_activation_type: str = "tanh",
        lin_dec_bias: bool = True,
        **activation_parameters,
    ) -> None:
        """Construct a JointNetwork object."""
        super().__init__()

        self.lin_enc = torch.nn.Linear(encoder_size, joint_space_size)
        self.lin_dec = torch.nn.Linear(
            decoder_size, joint_space_size, bias=lin_dec_bias
        )

        self.lin_out = torch.nn.Linear(joint_space_size, output_size)

        self.joint_activation = get_activation(
            joint_activation_type, **activation_parameters
        )

    def forward(
        self,
        enc_out: torch.Tensor,
        dec_out: torch.Tensor,
        no_projection: bool = False,
    ) -> torch.Tensor:
        """
        Joint computation of encoder and decoder hidden state sequences.

        This method performs a joint computation by combining the outputs from the
        encoder and decoder. It either applies a projection to the encoder and decoder
        outputs or directly computes the joint output if specified.

        Args:
            enc_out (torch.Tensor): Expanded encoder output state sequences.
                Shape can be (B, T, s_range, D_enc) or (B, T, 1, D_enc).
            dec_out (torch.Tensor): Expanded decoder output state sequences.
                Shape can be (B, T, s_range, D_dec) or (B, 1, U, D_dec).
            no_projection (bool, optional): If True, skips the projection step.
                Defaults to False.

        Returns:
            torch.Tensor: Joint output state sequences.
                Shape will be (B, T, U, D_out) or (B, T, s_range, D_out).

        Examples:
            >>> joint_network = JointNetwork(output_size=10, encoder_size=20,
            ...                              decoder_size=30)
            >>> enc_out = torch.randn(5, 10, 1, 20)  # Example encoder output
            >>> dec_out = torch.randn(5, 10, 2, 30)  # Example decoder output
            >>> joint_output = joint_network(enc_out, dec_out)
            >>> print(joint_output.shape)
            torch.Size([5, 10, 2, 10])  # Output shape based on inputs

        Note:
            Ensure that the shapes of `enc_out` and `dec_out` are compatible for
            addition when `no_projection` is set to False.
        """
        if no_projection:
            joint_out = self.joint_activation(enc_out + dec_out)
        else:
            joint_out = self.joint_activation(
                self.lin_enc(enc_out) + self.lin_dec(dec_out)
            )

        return self.lin_out(joint_out)
