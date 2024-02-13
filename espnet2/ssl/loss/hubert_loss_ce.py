# Adapted from torchaudio/wav2vec2/components.py - LogitGenerator
# Uses Cross-Entropy instead of HuBERTLoss - https://github.com/yanghaha0908/FastHuBERT/blob/master/criterion/fasthubert_criterion.py

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple, Union

from espnet2.ssl.loss.abs_loss import AbsLoss
from espnet.nets.pytorch_backend.nets_utils import th_accuracy

class HuBERTDecoder(nn.Module):
    """Generate the logits of masked and unmasked inputs.
    Args:
        encoder_embed_dim (int): The dimension of the transformer embedding output.
        num_classes (int): The number of classes in the labels.
        final_dim (int): Project final representations and targets to `final_dim`.
        skip_masked (bool): If True, skip computing losses over masked frames.
        skip_nomask (bool): If True, skip computing losses over unmasked frames.
    """

    def __init__(
        self,
        encoder_embed_dim: int,
        num_classes: int,
        final_dim: int,
        skip_masked: bool,
        skip_nomask: bool,
    ):
        super().__init__()
        self.final_proj = torch.nn.Linear(encoder_embed_dim, final_dim)
        self.label_embeddings = torch.nn.Linear(final_dim, num_classes)
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask

    def forward(self, x: Tensor, mask_m: Tensor, mask_u: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): The feature representation of the last transformer layer.
            label (Tensor): The label Tensor of dimension `[batch, frame]`.
            mask_m (Tensor): The masked indices of dimension `[batch, frame]`.
            mask_u (Tensor): The unmasked indices of dimension `[batch, frame]`.

        Returns:
            Tensor: The logits of masked frames. Tensor of dimension `[masked_frame, final_dim]`.
            Tensor: The logits of unmasked frames. Tensor of dimension `[unmasked_frame, final_dim]`.
        """
        logit_temp = 0.1
        proj_x = self.final_proj(x)

        logit_m = None
        logit_u = None
        if not self.skip_masked:
            proj_x_m = proj_x[mask_m]
            logit_m = self.label_embeddings(proj_x_m) / logit_temp

        if not self.skip_nomask:
            proj_x_u = proj_x[mask_u]
            logit_u = self.label_embeddings(proj_x_u) / logit_temp

        return logit_m, logit_u 

class HuBERTLossCrossEntropy(AbsLoss):
    def __init__(
        self,
        encoder_embed_dim: int,
        num_classes: int,
        final_dim: int,
        masked_weight: float = 1.0,
        unmasked_weight: float = 0.0,
        layers = [-1],
        layer_weights = [1.0],
    ):
        super().__init__()
        self.masked_loss_weight = masked_weight
        self.unmasked_loss_weight = unmasked_weight

        self.layers = layers
        self.layer_weights = layer_weights

        self.decoder = HuBERTDecoder(
            encoder_embed_dim,
            num_classes,
            final_dim,
            False,
            False,
        )

    def _compute_correct(
        self,
        logits,
        targets,
    ):
        if logits.numel() == 0:
            correct, count = 0, 0
        else:
            assert logits.dim() > 1, logits.shape
            max_idx = logits.argmax(-1)
            correct = (max_idx == targets).sum().item()
            count = max_idx.numel()
        return correct, count

    def forward(self, 
        xs_pad: torch.Tensor,
        ys_pad: torch.Tensor,
        mask_info,
        feature_penalty,
        feature_weight = 10,
    ):

        mask_m = mask_info["mask_m"]
        mask_u = mask_info["mask_u"]

        losses_m = []
        losses_u = []

        total_loss = 0.0
        targets_m = ys_pad[mask_m]
        targets_u = ys_pad[mask_u]
        for i, layer in enumerate(self.layers):
            x = xs_pad[i]
            hs_m, hs_u = self.decoder(x, mask_m, mask_u)

            if self.masked_loss_weight != 0.0:

                loss_m = F.cross_entropy(hs_m, targets_m.long(), reduction='sum') * self.masked_loss_weight * self.layer_weights[i]
                total_loss += loss_m
                divisor = hs_m.shape[0] if hs_m.shape[0] > 0 else 1
                losses_m.append(loss_m.detach().item() / divisor)

            if self.unmasked_loss_weight != 0.0:
                
                loss_u = F.cross_entropy(hs_u, targets_u.long(), reduction='sum') * self.unmasked_loss_weight * self.layer_weights[i]
                losses_u.append(loss_u.detach().item())
                total_loss += loss_u
                
        total_loss = feature_penalty * feature_weight * hs_m.shape[0] + total_loss

        correct_m, count_m = self._compute_correct(hs_m, targets_m)
        correct_u, count_u = self._compute_correct(hs_u, targets_u)

        #    hubert_losses_m=losses_m[0] if len(losses_m) > 0 else None, 
        #    hubert_losses_u=losses_u[0] if len(losses_u) > 0 else None,

        stats = dict(
            hubert_correct_m=correct_m,
            hubert_count_m=count_m,
            hubert_acc_m=0 if count_m == 0 else correct_m / count_m,
            hubert_correct_u=correct_u,
            hubert_count_u=count_u,
            hubert_acc_u=0 if count_u == 0 else correct_u / count_u,
        )

        for i, layer in enumerate(self.layers):
            stats[f'hubert_losses_m_{layer}'] = losses_m[i] if len(losses_m) > 0 else None
            stats[f'hubert_losses_u_{layer}'] = losses_u[i] if len(losses_u) > 0 else None

        return total_loss, stats

            

            
        

            

            

