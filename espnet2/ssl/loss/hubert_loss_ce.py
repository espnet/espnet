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
        final_dim: int,
        skip_masked: bool,
        skip_nomask: bool,
    ):
        super().__init__()
        self.final_proj = torch.nn.Linear(encoder_embed_dim, final_dim, bias=False)
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
        proj_x = self.final_proj(x)
        B, T, C = proj_x.shape
        #print("proj_x")
        #print(proj_x.shape)
        proj_x_m = None
        proj_x_u = None
        if not self.skip_masked:
            proj_x_m = proj_x[mask_m].reshape(B, -1, C)

        if not self.skip_nomask:
            proj_x_u = proj_x[mask_u]

        return proj_x_m, proj_x_u 

class HuBERTLoss(AbsLoss):
    def __init__(
        self,
        encoder_embed_dim: int,
        num_classes: int,
        final_dim: int,
        masked_loss: float = 1.0,
        unmasked_loss: float = 0.0,
        layers = [-1],
    ):
        super().__init__()
        self.masked_loss_weight = masked_loss
        self.unmasked_loss_weight = unmasked_loss

        self.layers = layers

        self.decoder = HuBERTDecoder(
            encoder_embed_dim,
            num_classes,
            True if masked_loss == 0.0 else False,
            True if unmasked_loss == 0.0 else False,
        )

    def forward(self, 
        xs_pad: torch.Tensor,
        ys_pad: torch.Tensor,
        mask_info,
        weight = 10
    ):

        mask_m = mask_info["mask_m"]
        mask_u = mask_info["mask_u"]

        losses_m = []
        losses_u = []

        accs_m = []
        accs_u = []
        total_loss = 0
        for layer in self.layers:
            x = xs_pad[layer]
            hs_m, hs_u = self.decoder(x, mask_m, mask_u)
            B, T, C = hs_m.shape

            if self.masked_loss_weight != 0.0:

                #print("xs_pad")
                #print(xs_pad[0].shape)
                #print("hs_m")
                #print(hs_m.shape, flush=True)
                #print("mask_m")
                #print(mask_m.shape, flush=True)
                #print("ys_pad")
                #print(ys_pad.shape, flush=True)
                #print("ys_pad_m")
                #print(ys_pad[mask_m].shape, flush=True)

                targets = ys_pad[mask_m].reshape(B, -1)
                loss_m = F.cross_entropy(hs_m.transpose(1,2), targets, reduction='sum', ignore_index=-1) * self.masked_loss_weight
                losses_m.append(loss_m.item())
                total_loss += loss_m

                acc_m = th_accuracy(
                    hs_m.view(B * T, C), # batch x seq x dim -> batch * seq x dim
                    targets, # batch x seq 
                    -1
                )

                accs_m.append(acc_m)

            if self.unmasked_loss_weight != 0.0:
                loss_u = F.cross_entropy(hs_u, ys_pad[mask_m], reduction='sum', ignore_index=-1) * self.unmasked_loss_weight
                losses_u.append(loss_u.item())
                total_loss += loss_u

            # maybe need to downweight loss here?
            # for now just return firs tloss item

        stats = {'hubert_losses_m': losses_m[0] if len(losses_m) > 0 else None, 'hubert_losses_u': losses_u[0] if len(losses_u) > 0 else None}
        stats['hubert_acc_m'] = accs_m[0] if len(losses_m) > 0 else None

        return total_loss, hs_m.shape[0] * weight, stats

            

            
        

            

            

