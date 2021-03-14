"""Auxiliary task implementation for transducer models."""

from itertools import chain
from typing import List
from typing import Union

import torch
import torch.nn.functional as F

from espnet.nets.transducer_decoder_interface import TransducerDecoderInterface


class AuxiliaryTask(torch.nn.Module):
    """Auxiliary task module."""

    def __init__(
        self,
        decoder: Union[torch.nn.Module, TransducerDecoderInterface],
        joint_network: torch.nn.Module,
        rnnt_criterion: torch.nn.Module,
        aux_task_type: str,
        aux_task_weight: int,
        encoder_out_dim: int,
        joint_dim: int,
    ):
        """Auxiliary task initialization.

        Args:
            decoder: Decoder module
            joint_network: Joint network module
            aux_task_type: Auxiliary task type
            aux_task_weight: Auxiliary task weight
            encoder_out: Encoder output dimension
            joint_dim: Joint space dimension

        """
        super().__init__()

        self.rnnt_criterion = rnnt_criterion

        self.mlp_net = torch.nn.Sequential(
            torch.nn.Linear(encoder_out_dim, joint_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(joint_dim, joint_dim),
        )

        self.decoder = decoder
        self.joint_network = joint_network

        self.aux_task_type = aux_task_type
        self.aux_task_weight = aux_task_weight

    def forward(
        self,
        enc_out_aux: List,
        dec_out: torch.Tensor,
        main_joint: torch.Tensor,
        target: torch.Tensor,
        pred_len: torch.Tensor,
        target_len: torch.Tensor,
    ) -> float:
        """Forward auxiliary task.

        Args:
            enc_out_aux: List of batch encoder intermediate outputs
                             [L x (B, T, D_enc_aux)]
            dec_out: Batch of ecoder outputs (B, U+1, D_dec)
            main_joint: Batch of joint output for main task
                           (B, T, U+1, vocab_size) or (sum(Tn * Un+1), vocab_size)
            target: Batch of target sequences (B, Lmax)
            pred_len: Batch of lengths of predicted sequences (B)
            target_len: Batch of lengths of target sequences (B)

        Returns:
            : Weighted auxiliary loss

        """
        aux_default = 0
        aux_symm_kl = 0

        if main_joint.dim() == 2:
            joint_memory_reduction = True
        else:
            joint_memory_reduction = False

        for p in chain(self.decoder.parameters(), self.joint_network.parameters()):
            p.requires_grad = False

        for i, enc_aux in enumerate(enc_out_aux):
            aux_mlp = self.mlp_net(enc_aux)

            aux_joint = self.joint_network(
                aux_mlp.unsqueeze(2),
                dec_out.unsqueeze(1),
                pred_len=pred_len if joint_memory_reduction else None,
                target_len=target_len if joint_memory_reduction else None,
                is_aux=True,
            )

            if self.aux_task_type != "symm_kl_div":
                aux_default += self.rnnt_criterion(
                    aux_joint,
                    target,
                    pred_len,
                    target_len,
                )

            if self.aux_task_type != "default":
                if joint_memory_reduction:
                    batch = target.size(0)
                    _start = 0

                    for b in range(batch):
                        t = int(pred_len[b])
                        u = int(target_len[b])
                        t_u = t * (u + 1)

                        main_b = main_joint[_start : (_start + t_u), :].view(
                            1, t, (u + 1), -1
                        )
                        aux_b = aux_joint[_start : (_start + t_u), :].view(
                            1, t, (u + 1), -1
                        )

                        aux_symm_kl += F.kl_div(
                            F.log_softmax(main_b, dim=-1),
                            F.softmax(aux_b, dim=-1),
                            reduction="mean",
                        ) + F.kl_div(
                            F.log_softmax(aux_b, dim=-1),
                            F.softmax(main_b, dim=-1),
                            reduction="mean",
                        )
                        _start += t_u

                    aux_symm_kl /= batch
                else:
                    aux_symm_kl += F.kl_div(
                        F.log_softmax(main_joint, dim=-1),
                        F.softmax(aux_joint, dim=-1),
                        reduction="mean",
                    ) + F.kl_div(
                        F.log_softmax(aux_joint, dim=-1),
                        F.softmax(main_joint, dim=-1),
                        reduction="mean",
                    )

        aux_main_loss = aux_default + aux_symm_kl

        for p in chain(self.decoder.parameters(), self.joint_network.parameters()):
            p.requires_grad = True

        return self.aux_task_weight * aux_main_loss
