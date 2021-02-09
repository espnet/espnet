"""Auxiliary task implementation for X-transducer models."""

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
        num_mlps: int,
        encoder_out_dim: int,
        joint_dim: int,
        output_dim: int,
        use_linear: bool = False,
    ):
        """Auxiliary task initialization.

        Args:
            decoder: Decoder module
            joint_network: Joint network module
            aux_task_type: Auxiliary task type
            aux_task_weight: Auxiliary task weight
            num_mlps: Number of auxiliary MLPs.
            encoder_out: Encoder output dimension
            joint_dim: Joint space dimension
            output_dim: Output dimension
            use_linear: Whether last encoder layer is an
                        auxiliary layer

        """
        super().__init__()

        self.rnnt_criterion = rnnt_criterion

        if aux_task_type == "jensen_shannon":
            self.kl_div = torch.nn.KLDivLoss(reduction="mean")
        elif aux_task_type == "cross_entropy":
            raise NotImplementedError

        self.mlp_net = torch.nn.Sequential(
            torch.nn.Linear(encoder_out_dim, joint_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(joint_dim, joint_dim),
        )

        if use_linear:
            self.lin = torch.nn.Linear(encoder_out_dim, joint_dim)
        self.use_linear = use_linear

        self.decoder = decoder
        self.joint_network = joint_network

        self.aux_task_type = aux_task_type
        self.aux_task_weight = aux_task_weight

    def forward(
        self,
        enc_out_aux: List,
        dec_out: torch.Tensor,
        joint_out: torch.Tensor,
        target: torch.Tensor,
        pred_len: torch.Tensor,
        target_len: torch.Tensor,
    ) -> float:
        """Forward auxiliary task.

        Args:
            enc_out_aux: List of encoder intermediate outputs
            dec_out: Decoder outputs
            joint_out: Joint output for main task
            target: Target labels
            pred_len: Prediction lengths
            target_len: Target lengths

        Returns:
            : Weighted auxiliary loss

        """
        aux_main_loss = 0
        len_aux = len(enc_out_aux) - 1

        for p in chain(self.decoder.parameters(), self.joint_network.parameters()):
            p.requires_grad = False

        for i, enc_aux in enumerate(enc_out_aux):
            if i == len_aux and self.use_linear:
                aux_lin = self.lin(enc_aux)
            else:
                aux_lin = self.mlp_net(enc_aux)

            aux_joint = self.joint_network(
                aux_lin.unsqueeze(2),
                dec_out.unsqueeze(1),
                is_aux=True,
            )

            aux_loss = self.rnnt_criterion(
                aux_joint,
                target,
                pred_len,
                target_len,
            )

            if self.aux_task_type == "jensen_shannon":
                M = 0.5 * (
                    torch.softmax(joint_out, dim=-1) + torch.softmax(aux_joint, dim=-1)
                )

                js_div_loss = 0.5 * (
                    self.kl_div(F.log_softmax(joint_out, dim=-1), M)
                    + self.kl_div(F.log_softmax(aux_joint, dim=-1), M)
                )

                aux_main_loss += aux_loss + js_div_loss
            else:
                aux_main_loss += aux_loss

        for p in chain(self.decoder.parameters(), self.joint_network.parameters()):
            p.requires_grad = True

        return self.aux_task_weight * aux_main_loss
