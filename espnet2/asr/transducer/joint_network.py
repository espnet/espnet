"""Transducer joint network implementation."""

import torch

from espnet.nets.pytorch_backend.nets_utils import get_activation


class JointNetwork(torch.nn.Module):
    """Transducer joint network module.

    Args:
        joint_output_size: Joint network output dimension
        encoder_output_size: Encoder output dimension.
        decoder_output_size: Decoder output dimension.
        joint_space_size: Dimension of joint space.
        joint_activation_type: Type of activation for joint network.

    """

    def __init__(
        self,
        joint_output_size: int,
        encoder_output_size: int,
        decoder_output_size: int,
        joint_space_size: int = 256,
        joint_activation_type: str = "tanh",
    ):
        """Joint network initializer."""
        super().__init__()

        self.lin_enc = torch.nn.Linear(encoder_output_size, joint_space_size)
        self.lin_dec = torch.nn.Linear(decoder_output_size, joint_space_size)

        self.lin_out = torch.nn.Linear(joint_space_size, joint_output_size)

        self.joint_activation = get_activation(joint_activation_type)

    def forward(
        self,
        enc_out: torch.Tensor,
        dec_out: torch.Tensor,
    ) -> torch.Tensor:
        """Joint computation of encoder and decoder hidden state sequences.

        Args:
            enc_out: Expanded encoder output state sequences (B, T, 1, D_enc)
            dec_out: Expanded decoder output state sequences (B, 1, U, D_dec)

        Returns:
            joint_out: Joint output state sequences. (B, T, U, D_out)

        """
        joint_out = self.joint_activation(self.lin_enc(enc_out) + self.lin_dec(dec_out))

        return self.lin_out(joint_out)
