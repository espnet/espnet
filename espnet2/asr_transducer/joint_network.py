"""Transducer joint network implementation."""

import torch

from espnet2.asr_transducer.activation import get_activation


class JointNetwork(torch.nn.Module):
    """Transducer joint network module.

    Args:
        dim_output: Output dimension
        dim_encoder: Encoder output dimension.
        dim_decoder: Decoder output dimension.
        dim_joint_space: Dimension of joint space.
        joint_act_type: Type of activation for joint network.

    """

    def __init__(
        self,
        dim_output: int,
        dim_encoder: int,
        dim_decoder: int,
        dim_joint_space: int = 256,
        joint_activation_type: str = "tanh",
        **activation_parameters,
    ):
        """Joint network initializer."""
        super().__init__()

        self.lin_enc = torch.nn.Linear(dim_encoder, dim_joint_space)
        self.lin_dec = torch.nn.Linear(dim_decoder, dim_joint_space, bias=False)

        self.lin_out = torch.nn.Linear(dim_joint_space, dim_output)

        self.joint_activation = get_activation(
            joint_activation_type, **activation_parameters
        )

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
