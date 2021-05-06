"""Transducer joint network implementation."""

import torch

from espnet.nets.pytorch_backend.nets_utils import get_activation


class JointNetwork(torch.nn.Module):
    """Transducer joint network module.

    Args:
        joint_space_size: Dimension of joint space
        joint_activation_type: Activation type for joint network

    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        decoder_output_size: int,
        joint_space_size: int,
        joint_activation_type: int,
    ):
        """Joint network initializer."""
        super().__init__()

        self.lin_enc = torch.nn.Linear(encoder_output_size, joint_space_size)
        self.lin_dec = torch.nn.Linear(
            decoder_output_size, joint_space_size, bias=False
        )

        self.lin_out = torch.nn.Linear(joint_space_size, vocab_size)

        self.joint_activation = get_activation(joint_activation_type)

    def forward(
        self, h_enc: torch.Tensor, h_dec: torch.Tensor, is_aux: bool = False
    ) -> torch.Tensor:
        """Joint computation of z.

        Args:
            h_enc: Batch of expanded hidden state (B, T, 1, D_enc)
            h_dec: Batch of expanded hidden state (B, 1, U, D_dec)

        Returns:
            z: Output (B, T, U, vocab_size)

        """
        if is_aux:
            z = self.joint_activation(h_enc + self.lin_dec(h_dec))
        else:
            z = self.joint_activation(self.lin_enc(h_enc) + self.lin_dec(h_dec))
        z = self.lin_out(z)

        return z
