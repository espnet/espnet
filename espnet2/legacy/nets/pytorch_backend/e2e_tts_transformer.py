# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""TTS-Transformer related modules."""


import torch

from espnet2.legacy.nets.pytorch_backend.e2e_tts_tacotron2 import (
    GuidedAttentionLoss,
)
from espnet2.legacy.nets.pytorch_backend.e2e_tts_tacotron2 import (  # noqa: F401
    Tacotron2Loss as TransformerLoss,
)


class GuidedMultiHeadAttentionLoss(GuidedAttentionLoss):
    """Guided attention loss function module for multi head attention.

    Args:
        sigma (float, optional): Standard deviation to control
        how close attention to a diagonal.
        alpha (float, optional): Scaling coefficient (lambda).
        reset_always (bool, optional): Whether to always reset masks.

    """

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.

        Args:
            att_ws (Tensor):
                Batch of multi head attention weights (B, H, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lengths (B,).
            olens (LongTensor): Batch of output lengths (B,).

        Returns:
            Tensor: Guided attention loss value.

        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = (
                self._make_guided_attention_masks(ilens, olens)
                .to(att_ws.device)
                .unsqueeze(1)
            )
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device).unsqueeze(1)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()

        return self.alpha * loss
