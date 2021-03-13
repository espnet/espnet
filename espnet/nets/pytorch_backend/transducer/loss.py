#!/usr/bin/env python3

"""Transducer loss module."""

import torch


class TransLoss(torch.nn.Module):
    """Transducer loss module.

    Args:
        trans_type: Type of transducer implementation to calculate loss.
        joint_memory_reduction: Whether joint memory reduction is used.
        blank_id: Blank symbol ID.

    """

    def __init__(self, trans_type: str, joint_memory_reduction: bool, blank_id: int):
        """Construct an TransLoss object."""
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if trans_type == "warp-transducer":
            from warprnnt_pytorch import RNNTLoss

            self.trans_loss = RNNTLoss(blank=blank_id)
        elif trans_type == "warp-rnnt":
            if device.type == "cuda":
                try:
                    from warp_rnnt import rnnt_loss

                    self.trans_loss = rnnt_loss
                except ImportError:
                    raise ImportError(
                        "warp-rnnt is not installed. Please re-setup"
                        " espnet or use 'warp-transducer'"
                    )
            else:
                raise ValueError("warp-rnnt is not supported in CPU mode")

        self.trans_type = trans_type
        self.joint_memory_reduction = joint_memory_reduction
        self.blank_id = blank_id

    def forward(
        self,
        pred_pad: torch.Tensor,
        target: torch.Tensor,
        pred_len: torch.Tensor,
        target_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute path-aware regularization transducer loss.

        Args:
            pred_pad: Batch of predicted sequences
                (B, T, U+1, odim) or
                (sum(Tn * (U+1)n), odim)
            target: Batch of target sequences (B, T)
            pred_len: Batch of lengths of predicted sequences (B)
            target_len: Batch of lengths of target sequences (B)

        Returns:
            loss: transducer loss

        """
        dtype = pred_pad.dtype

        if dtype != torch.float32:
            # warp-transducer and warp-rnnt only support float32
            pred_pad = pred_pad.to(dtype=torch.float32)

        if self.joint_memory_reduction:
            batch = target.size(0)
            loss = torch.zeros((1), dtype=torch.float32, device=pred_pad.device)
            _start = 0

            for b in range(batch):
                t = int(pred_len[b])
                u = int(target_len[b])
                t_u = t * (u + 1)

                if self.trans_type == "warp-rnnt":
                    log_probs = torch.log_softmax(
                        pred_pad[_start : (_start + t_u), :].view(1, t, (u + 1), -1),
                        dim=-1,
                    )

                    loss += self.trans_loss(
                        log_probs,
                        target[b : (b + 1), :u],
                        pred_len[b].unsqueeze(0),
                        target_len[b].unsqueeze(0),
                        reduction="mean",
                        blank=self.blank_id,
                        gather=True,
                    )
                else:
                    loss += self.trans_loss(
                        pred_pad[_start : (_start + t_u), :].view(1, t, (u + 1), -1),
                        target[b : (b + 1), :u],
                        pred_len[b].unsqueeze(0),
                        target_len[b].unsqueeze(0),
                    )

                _start += t_u

            loss /= batch
        else:
            if self.trans_type == "warp-rnnt":
                log_probs = torch.log_softmax(pred_pad, dim=-1)

                loss = self.trans_loss(
                    log_probs,
                    target,
                    pred_len,
                    target_len,
                    reduction="mean",
                    blank=self.blank_id,
                    gather=True,
                )
            else:
                loss = self.trans_loss(pred_pad, target, pred_len, target_len)

        loss = loss.to(dtype=dtype)

        return loss
