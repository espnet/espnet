"""Transducer joint network implementation."""

from typing import Optional

import torch

from espnet.nets.pytorch_backend.nets_utils import get_activation


class JointNetwork(torch.nn.Module):
    """Transducer joint network module.

    Args:
        vocab_size: Size of vocabulary
        encoder_output_size: Dimension of encoder output
        decoder_output_size: Dimension of decoder output
        joint_space_size: Dimension of joint space
        joint_activation_type: Activation type for joint network
        joint_memory_reduction: Whether to use efficient joint computation
            Based on https://microsoft.com/en-us/research/uploads/prod/2019/10/RNNT.pdf

    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        decoder_output_size: int,
        joint_space_size: int,
        joint_activation_type: int,
        joint_memory_reduction: bool,
    ):
        """Joint network initializer."""
        super().__init__()

        self.lin_enc = torch.nn.Linear(encoder_output_size, joint_space_size)
        self.lin_dec = torch.nn.Linear(
            decoder_output_size, joint_space_size, bias=False
        )

        self.lin_out = torch.nn.Linear(joint_space_size, vocab_size)

        self.joint_activation = get_activation(joint_activation_type)

        self.joint_memory_reduction = joint_memory_reduction
        self.joint_dim = joint_space_size

    def forward(
        self,
        h_enc: torch.Tensor,
        h_dec: torch.Tensor,
        is_aux: bool = False,
        pred_len: Optional[torch.Tensor] = None,
        target_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Joint computation of z.

        Args:
            h_enc: Batch of expanded hidden state (B, T, 1, D_enc)
            h_dec: Batch of expanded hidden state (B, 1, U, D_dec)
            is_aux: Whether auxiliary encoder representation is provided
            pred_len: batch of lengths of predicted sequences (B)
            target_len: batch of lengths of target sequences (B)

        Returns:
            z: Output (B, T, U, vocab_size) or (sum(Tn * Un), vocab_size)

        """
        if self.joint_memory_reduction:
            batch = h_dec.size(0)

            z = h_dec.new_zeros((sum(pred_len * (target_len + 1)), self.joint_dim))
            _start = 0

            for b in range(batch):
                t = int(pred_len[b])
                u_1 = int(target_len[b]) + 1
                t_u = t * u_1

                if is_aux:
                    z[_start : (_start + t_u), :] = self.joint_activation(
                        h_enc[b][:t, :, :] + self.lin_dec(h_dec[b][:, :u_1, :])
                    ).view(t_u, -1)
                else:
                    z[_start : (_start + t_u), :] = self.joint_activation(
                        self.lin_enc(h_enc[b][:t, :, :])
                        + self.lin_dec(h_dec[b][:, :u_1, :])
                    ).view(t_u, -1)

                _start += t_u
        else:
            if is_aux:
                z = self.joint_activation(h_enc + self.lin_dec(h_dec))
            else:
                z = self.joint_activation(self.lin_enc(h_enc) + self.lin_dec(h_dec))

        z = self.lin_out(z)

        return z
