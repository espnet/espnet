"""Module implementing Transducer main and auxiliary tasks."""

from typing import Any, List, Optional, Tuple

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)


class TransducerTasks(torch.nn.Module):
    """Transducer tasks module."""

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joint_dim: int,
        output_dim: int,
        joint_activation_type: str = "tanh",
        transducer_loss_weight: float = 1.0,
        ctc_loss: bool = False,
        ctc_loss_weight: float = 0.5,
        ctc_loss_dropout_rate: float = 0.0,
        lm_loss: bool = False,
        lm_loss_weight: float = 0.5,
        lm_loss_smoothing_rate: float = 0.0,
        aux_transducer_loss: bool = False,
        aux_transducer_loss_weight: float = 0.2,
        aux_transducer_loss_mlp_dim: int = 320,
        aux_trans_loss_mlp_dropout_rate: float = 0.0,
        symm_kl_div_loss: bool = False,
        symm_kl_div_loss_weight: float = 0.2,
        fastemit_lambda: float = 0.0,
        blank_id: int = 0,
        ignore_id: int = -1,
        training: bool = False,
    ):
        """Initialize module for Transducer tasks.

        Args:
            encoder_dim: Encoder outputs dimension.
            decoder_dim: Decoder outputs dimension.
            joint_dim: Joint space dimension.
            output_dim: Output dimension.
            joint_activation_type: Type of activation for joint network.
            transducer_loss_weight: Weight for main transducer loss.
            ctc_loss: Compute CTC loss.
            ctc_loss_weight: Weight of CTC loss.
            ctc_loss_dropout_rate: Dropout rate for CTC loss inputs.
            lm_loss: Compute LM loss.
            lm_loss_weight: Weight of LM loss.
            lm_loss_smoothing_rate: Smoothing rate for LM loss' label smoothing.
            aux_transducer_loss: Compute auxiliary transducer loss.
            aux_transducer_loss_weight: Weight of auxiliary transducer loss.
            aux_transducer_loss_mlp_dim: Hidden dimension for aux. transducer MLP.
            aux_trans_loss_mlp_dropout_rate: Dropout rate for aux. transducer MLP.
            symm_kl_div_loss: Compute KL divergence loss.
            symm_kl_div_loss_weight: Weight of KL divergence loss.
            fastemit_lambda: Regularization parameter for FastEmit.
            blank_id: Blank symbol ID.
            ignore_id: Padding symbol ID.
            training: Whether the model was initializated in training or inference mode.

        """
        super().__init__()

        if not training:
            ctc_loss, lm_loss, aux_transducer_loss, symm_kl_div_loss = (
                False,
                False,
                False,
                False,
            )

        self.joint_network = JointNetwork(
            output_dim, encoder_dim, decoder_dim, joint_dim, joint_activation_type
        )

        if training:
            from warprnnt_pytorch import RNNTLoss

            self.transducer_loss = RNNTLoss(
                blank=blank_id, reduction="sum", fastemit_lambda=fastemit_lambda,
            )

        if ctc_loss:
            self.ctc_lin = torch.nn.Linear(encoder_dim, output_dim)

            self.ctc_loss = torch.nn.CTCLoss(
                blank=blank_id, reduction="none", zero_infinity=True,
            )

        if aux_transducer_loss:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(encoder_dim, aux_transducer_loss_mlp_dim),
                torch.nn.LayerNorm(aux_transducer_loss_mlp_dim),
                torch.nn.Dropout(p=aux_trans_loss_mlp_dropout_rate),
                torch.nn.ReLU(),
                torch.nn.Linear(aux_transducer_loss_mlp_dim, joint_dim),
            )

            if symm_kl_div_loss:
                self.kl_div = torch.nn.KLDivLoss(reduction="sum")

        if lm_loss:
            self.lm_lin = torch.nn.Linear(decoder_dim, output_dim)

            self.label_smoothing_loss = LabelSmoothingLoss(
                output_dim, ignore_id, lm_loss_smoothing_rate, normalize_length=False
            )

        self.output_dim = output_dim

        self.transducer_loss_weight = transducer_loss_weight

        self.use_ctc_loss = ctc_loss
        self.ctc_loss_weight = ctc_loss_weight
        self.ctc_dropout_rate = ctc_loss_dropout_rate

        self.use_lm_loss = lm_loss
        self.lm_loss_weight = lm_loss_weight

        self.use_aux_transducer_loss = aux_transducer_loss
        self.aux_transducer_loss_weight = aux_transducer_loss_weight

        self.use_symm_kl_div_loss = symm_kl_div_loss
        self.symm_kl_div_loss_weight = symm_kl_div_loss_weight

        self.blank_id = blank_id
        self.ignore_id = ignore_id

        self.target = None

    def compute_transducer_loss(
        self,
        enc_out: torch.Tensor,
        dec_out: torch.tensor,
        target: torch.Tensor,
        t_len: torch.Tensor,
        u_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.

        Args:
            enc_out: Encoder output sequences. (B, T, D_enc)
            dec_out: Decoder output sequences. (B, U, D_dec)
            target: Target label ID sequences. (B, L)
            t_len: Time lengths. (B,)
            u_len: Label lengths. (B,)

        Returns:
            (joint_out, loss_trans):
                Joint output sequences. (B, T, U, D_joint),
                Transducer loss value.

        """
        joint_out = self.joint_network(enc_out.unsqueeze(2), dec_out.unsqueeze(1))

        loss_trans = self.transducer_loss(joint_out, target, t_len, u_len)
        loss_trans /= joint_out.size(0)

        return joint_out, loss_trans

    def compute_ctc_loss(
        self,
        enc_out: torch.Tensor,
        target: torch.Tensor,
        t_len: torch.Tensor,
        u_len: torch.Tensor,
    ):
        """Compute CTC loss.

        Args:
            enc_out: Encoder output sequences. (B, T, D_enc)
            target: Target character ID sequences. (B, U)
            t_len: Time lengths. (B,)
            u_len: Label lengths. (B,)

        Returns:
            : CTC loss value.

        """
        ctc_lin = self.ctc_lin(
            torch.nn.functional.dropout(
                enc_out.to(dtype=torch.float32), p=self.ctc_dropout_rate
            )
        )
        ctc_logp = torch.log_softmax(ctc_lin.transpose(0, 1), dim=-1)

        with torch.backends.cudnn.flags(deterministic=True):
            loss_ctc = self.ctc_loss(ctc_logp, target, t_len, u_len)

        return loss_ctc.mean()

    def compute_aux_transducer_and_symm_kl_div_losses(
        self,
        aux_enc_out: torch.Tensor,
        dec_out: torch.Tensor,
        joint_out: torch.Tensor,
        target: torch.Tensor,
        aux_t_len: torch.Tensor,
        u_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute auxiliary Transducer loss and Jensen-Shannon divergence loss.

        Args:
            aux_enc_out: Encoder auxiliary output sequences. [N x (B, T_aux, D_enc_aux)]
            dec_out: Decoder output sequences. (B, U, D_dec)
            joint_out: Joint output sequences. (B, T, U, D_joint)
            target: Target character ID sequences. (B, L)
            aux_t_len: Auxiliary time lengths. [N x (B,)]
            u_len: True U lengths. (B,)

        Returns:
           : Auxiliary Transducer loss and KL divergence loss values.

        """
        aux_trans_loss = 0
        symm_kl_div_loss = 0

        num_aux_layers = len(aux_enc_out)
        B, T, U, D = joint_out.shape

        for p in self.joint_network.parameters():
            p.requires_grad = False

        for i, aux_enc_out_i in enumerate(aux_enc_out):
            aux_mlp = self.mlp(aux_enc_out_i)

            aux_joint_out = self.joint_network(
                aux_mlp.unsqueeze(2), dec_out.unsqueeze(1), is_aux=True,
            )

            if self.use_aux_transducer_loss:
                aux_trans_loss += (
                    self.transducer_loss(aux_joint_out, target, aux_t_len[i], u_len,)
                    / B
                )

            if self.use_symm_kl_div_loss:
                denom = B * T * U

                kl_main_aux = (
                    self.kl_div(
                        torch.log_softmax(joint_out, dim=-1),
                        torch.softmax(aux_joint_out, dim=-1),
                    )
                    / denom
                )

                kl_aux_main = (
                    self.kl_div(
                        torch.log_softmax(aux_joint_out, dim=-1),
                        torch.softmax(joint_out, dim=-1),
                    )
                    / denom
                )

                symm_kl_div_loss += kl_main_aux + kl_aux_main

        for p in self.joint_network.parameters():
            p.requires_grad = True

        aux_trans_loss /= num_aux_layers

        if self.use_symm_kl_div_loss:
            symm_kl_div_loss /= num_aux_layers

        return aux_trans_loss, symm_kl_div_loss

    def compute_lm_loss(
        self, dec_out: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """Forward LM loss.

        Args:
            dec_out: Decoder output sequences. (B, U, D_dec)
            target: Target label ID sequences. (B, U)

        Returns:
            : LM loss value.

        """
        lm_lin = self.lm_lin(dec_out)

        lm_loss = self.label_smoothing_loss(lm_lin, target)

        return lm_loss

    def set_target(self, target: torch.Tensor):
        """Set target label ID sequences.

        Args:
            target: Target label ID sequences. (B, L)

        """
        self.target = target

    def get_target(self):
        """Set target label ID sequences.

        Args:

        Returns:
            target: Target label ID sequences. (B, L)

        """
        return self.target

    def get_transducer_tasks_io(
        self,
        labels: torch.Tensor,
        enc_out_len: torch.Tensor,
        aux_enc_out_len: Optional[List],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get Transducer tasks inputs and outputs.

        Args:
            labels: Label ID sequences. (B, U)
            enc_out_len: Time lengths. (B,)
            aux_enc_out_len: Auxiliary time lengths. [N X (B,)]

        Returns:
            target: Target label ID sequences. (B, L)
            lm_loss_target: LM loss target label ID sequences. (B, U)
            t_len: Time lengths. (B,)
            aux_t_len: Auxiliary time lengths. [N x (B,)]
            u_len: Label lengths. (B,)

        """
        device = labels.device

        labels_unpad = [label[label != self.ignore_id] for label in labels]
        blank = labels[0].new([self.blank_id])

        target = pad_list(labels_unpad, self.blank_id).type(torch.int32).to(device)
        lm_loss_target = (
            pad_list(
                [torch.cat([y, blank], dim=0) for y in labels_unpad], self.ignore_id
            )
            .type(torch.int64)
            .to(device)
        )

        self.set_target(target)

        if enc_out_len.dim() > 1:
            enc_mask_unpad = [m[m != 0] for m in enc_out_len]
            enc_out_len = list(map(int, [m.size(0) for m in enc_mask_unpad]))
        else:
            enc_out_len = list(map(int, enc_out_len))

        t_len = torch.IntTensor(enc_out_len).to(device)
        u_len = torch.IntTensor([label.size(0) for label in labels_unpad]).to(device)

        if aux_enc_out_len:
            aux_t_len = []

            for i in range(len(aux_enc_out_len)):
                if aux_enc_out_len[i].dim() > 1:
                    aux_mask_unpad = [aux[aux != 0] for aux in aux_enc_out_len[i]]
                    aux_t_len.append(
                        torch.IntTensor(
                            list(map(int, [aux.size(0) for aux in aux_mask_unpad]))
                        ).to(device)
                    )
                else:
                    aux_t_len.append(
                        torch.IntTensor(list(map(int, aux_enc_out_len[i]))).to(device)
                    )
        else:
            aux_t_len = aux_enc_out_len

        return target, lm_loss_target, t_len, aux_t_len, u_len

    def forward(
        self,
        enc_out: torch.Tensor,
        aux_enc_out: List[torch.Tensor],
        dec_out: torch.Tensor,
        labels: torch.Tensor,
        enc_out_len: torch.Tensor,
        aux_enc_out_len: torch.Tensor,
    ) -> Tuple[Tuple[Any], float, float]:
        """Forward main and auxiliary task.

        Args:
            enc_out: Encoder output sequences. (B, T, D_enc)
            aux_enc_out: Encoder intermediate output sequences. (B, T_aux, D_enc_aux)
            dec_out: Decoder output sequences. (B, U, D_dec)
            target: Target label ID sequences. (B, L)
            t_len: Time lengths. (B,)
            aux_t_len: Auxiliary time lengths. (B,)
            u_len: Label lengths. (B,)

        Returns:
            : Weighted losses.
              (transducer loss, ctc loss, aux Transducer loss, KL div loss, LM loss)
            cer: Sentence-level CER score.
            wer: Sentence-level WER score.

        """
        if self.use_symm_kl_div_loss:
            assert self.use_aux_transducer_loss

        (trans_loss, ctc_loss, lm_loss, aux_trans_loss, symm_kl_div_loss) = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        target, lm_loss_target, t_len, aux_t_len, u_len = self.get_transducer_tasks_io(
            labels, enc_out_len, aux_enc_out_len,
        )

        joint_out, trans_loss = self.compute_transducer_loss(
            enc_out, dec_out, target, t_len, u_len
        )

        if self.use_ctc_loss:
            ctc_loss = self.compute_ctc_loss(enc_out, target, t_len, u_len)

        if self.use_aux_transducer_loss:
            (
                aux_trans_loss,
                symm_kl_div_loss,
            ) = self.compute_aux_transducer_and_symm_kl_div_losses(
                aux_enc_out, dec_out, joint_out, target, aux_t_len, u_len,
            )

        if self.use_lm_loss:
            lm_loss = self.compute_lm_loss(dec_out, lm_loss_target)

        return (
            self.transducer_loss_weight * trans_loss,
            self.ctc_loss_weight * ctc_loss,
            self.aux_transducer_loss_weight * aux_trans_loss,
            self.symm_kl_div_loss_weight * symm_kl_div_loss,
            self.lm_loss_weight * lm_loss,
        )
