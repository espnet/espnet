from torch import nn

from longformer.longformer import LongformerSelfAttention,LongformerConfig


class LongformerAttention(nn.Module):
    def __init__(self, config:LongformerConfig, layer_id:int):
        """Longformer Self-Attention Wrapper.
        
        Args:
            config : Longformer attention configuration 
            layer_id: Integer representing the layer index
        """
        assert check_argument_types()
        super().__init__()
        self.attention_window = config.attention_window[layer_id]
        self.attention_layer = LongformerSelfAttention(config, layer_id=layer_id)
        self.attention = None

    def forward(self, query, key, value, mask):
        """ Computes Longformer Self-Attention with masking.
       
        Expects `len(hidden_states)` to be multiple of `attention_window`.Padding to `attention_window` happens in :meth:`encoder.forward` to avoid redoing the padding on each layer.
        The `attention_mask` is changed in :meth:`LongformerAttention.forward`
        from 0, 1, 2 to:
            * -1: no attention
            * 0: local attention
            * +1: global attention

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        attention_mask = mask.int()
        attention_mask[mask == 0] = -1
        attention_mask[mask == 1] = 0
        output, self.attention = self.attention_layer(
            hidden_states=query,
            attention_mask=attention_mask.unsqueeze(1),
            head_mask=None,
            output_attentions=True,
        )
        return output
