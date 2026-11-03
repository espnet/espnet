# Copyright 2025 Qingzheng Wang
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Residual lang2vec conditioning around an unchanged SSL encoder layer."""

import torch
from torch import nn


class Lang2VecConditionedLayer(nn.Module):
    """Predict lang2vec after an upstream layer and add its projected residual.

    The original layer owns attention and checkpointing. The LID model consumes
    ``intermediate_out`` for the auxiliary loss and clears it after each call.
    As with self-conditioned interCTC, the model creates the projection, which
    may be shared across several wrappers.
    """

    def __init__(
        self,
        layer,
        encoder,
        pooling,
        projector,
        prediction,
        conditioning_layer=None,
        detach_prediction=False,
        detach_condition=False,
    ):
        super().__init__()
        self.layer = layer
        self.encoder = encoder
        self.pooling = pooling
        self.projector = projector
        self.prediction = prediction
        self.conditioning_layer = conditioning_layer
        self.detach_prediction = detach_prediction
        self.detach_condition = detach_condition
        self.intermediate_out = None

    def forward(
        self, hidden_states, attention_mask=None, output_attentions=False, **kwargs
    ):
        """Preserve the upstream output tuple and pass attention arguments through."""
        outputs = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = outputs[0]
        lengths = None
        if attention_mask is not None:
            # Consume the stock upstream's mask; never build or alter it.
            mask = attention_mask
            if mask.ndim == 4:
                mask = mask[:, 0, 0, :]
            if mask.ndim != 2:
                raise ValueError("Expected a 2-D or 4-D SSL attention mask.")
            valid = mask if mask.dtype == torch.bool else mask == 0
            lengths = valid.long().sum(-1)
        frames = self.encoder(hidden_states)
        embedding = self.projector(self.pooling(frames, feat_lengths=lengths))
        prediction = self.prediction(embedding)
        self.intermediate_out = prediction
        if self.conditioning_layer is not None:
            value = prediction.detach() if self.detach_prediction else prediction
            condition = self.conditioning_layer(value).unsqueeze(1)
            if self.detach_condition:
                condition = condition.detach()
            hidden_states = hidden_states + condition
        return (hidden_states,) + outputs[1:]
