"""Auxiliary task implementation for transducer models."""

from itertools import chain
from typing import List
from typing import Tuple
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward auxiliary task.

        Args:
            enc_out_aux: List of encoder intermediate outputs
            dec_out: Decoder outputs
            main_joint: Joint output for main task
            target: Target labels
            pred_len: Prediction lengths
            target_len: Target lengths

        Returns:
            : (Weighted auxiliary transducer loss, Weighted auxiliary symmetric KL loss)

        """
        aux_trans = 0
        aux_symm_kl = 0

        for p in chain(self.decoder.parameters(), self.joint_network.parameters()):
            p.requires_grad = False

        for i, enc_aux in enumerate(enc_out_aux):
            aux_mlp = self.mlp_net(enc_aux)

            aux_joint = self.joint_network(
                aux_mlp.unsqueeze(2),
                dec_out.unsqueeze(1),
                is_aux=True,
            )

            if self.aux_task_type != "symm_kl_div":
                aux_trans += self.rnnt_criterion(
                    aux_joint,
                    target,
                    pred_len,
                    target_len,
                )

            if self.aux_task_type != "default":
                aux_symm_kl += F.kl_div(
                    F.log_softmax(main_joint, dim=-1),
                    F.softmax(aux_joint, dim=-1),
                    reduction="mean",
                ) + F.kl_div(
                    F.log_softmax(aux_joint, dim=-1),
                    F.softmax(main_joint, dim=-1),
                    reduction="mean",
                )

        for p in chain(self.decoder.parameters(), self.joint_network.parameters()):
            p.requires_grad = True

        return self.aux_task_weight * aux_trans, self.aux_task_weight * aux_symm_kl
