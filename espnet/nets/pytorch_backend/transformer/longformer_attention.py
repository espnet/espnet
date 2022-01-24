import torch
from torch import nn

from longformer.longformer import LongformerSelfAttention


class LongformerAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()

        self.attention_window = config.attention_window[layer_id]

        self.attention_layer = LongformerSelfAttention(config, layer_id=layer_id)
        self.attention = None

    def forward(self, query, key, value, mask, pos_emb=None):
        """
        :class:`LongformerSelfAttention` expects `len(hidden_states)` to be multiple of `attention_window`. Padding to
        `attention_window` happens in :meth:`LongformerModel.forward` to avoid redoing the padding on each layer.
        The `attention_mask` is changed in :meth:`LongformerModel.forward` from 0, 1, 2 to:
            * -10000: no attention
            * 0: local attention
            * +10000: global attention
        """
        attention_mask = mask.int()
        attention_mask[mask == False] = -1
        attention_mask[mask == True] = 0
        output, self.attention = self.attention_layer(
            hidden_states=query,
            attention_mask=attention_mask.unsqueeze(1),
            head_mask=None,
            output_attentions=True,
        )
        # output, self.attention = self.attention_layer(
        #     hidden_states=query,
        #     attention_mask=attention_mask.unsqueeze(1),
        #     output_attentions=True,
        # )
        return output

        ##
