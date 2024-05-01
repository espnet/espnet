import torch
import math
from packaging.version import parse as V

from typing import Dict


class MultiHeadedAttention(torch.nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate, causal=False, flashattention=True):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = torch.nn.Linear(n_feat, n_feat)
        self.linear_k = torch.nn.Linear(n_feat, n_feat)
        self.linear_v = torch.nn.Linear(n_feat, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)

        self.dropout_rate = dropout_rate
        self.causal = causal
        self.flashattention = flashattention

        # check flashattention availability
        torch_version = torch.__version__
        if flashattention and V(torch_version) < V("2.0.1"):
            raise ValueError(f"Upgrade Pytorch 2.0.1+ to use flashattention")

    def forward(self, query, key, value, mask, cache):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            cache (dict): with format [torch.nn.Module: Tensor], 
                with tensor size (#batch, time1/time2, size),
                the KV-Cache.

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        Hint: len(cache) > 0 is only applied when the attention is
            causal and for auto-regressive inference.
        """

        if self.causal and mask is not None:
            raise ValueError("Cannot require causality when mask is provided.")
        
        assert isinstance(cache, dict)
        if len(cache) > 0 and not self.causal:
            raise ValueError("Cache is only used in causal attention")
        
        causal = self.causal if self.linear_k not in cache else False
        
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)

        if len(cache) > 0:
            _ = self.linear_k(key) # accumulate result in cache
            _ = self.linear_v(value)
            k = cache[self.linear_k].view(n_batch, -1, self.h, self.d_k)
            v = cache[self.linear_v].view(n_batch, -1, self.h, self.d_k)
        else:
            k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
            v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        
        x = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask.bool().unsqueeze(1) if mask is not None else None,
            dropout_p=self.dropout_rate,
            is_causal=causal,
        ).transpose(1, 2).flatten(2, 3)

        return self.linear_out(x)

class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)

class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, cache: dict = None):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        if cache is not None and len(cache) > 0: # Inference time
            entry = next(iter(cache.values()))
            assert isinstance(entry, torch.Tensor) and entry.dim() == 3
            start = entry.size(1)
        else:
            start = 0
        x = x * self.xscale + self.pe[:, start: start + x.size(1)]
        return self.dropout(x)

class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        att_unit: int = 256,
        head: int = 2,
        unit: int = 1024,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        causal: bool = True,
        flashattention: bool = False,
        cross_attention: bool = False,
    ):
        super(TransformerLayer, self).__init__()

        self.attn = MultiHeadedAttention(
            head,
            att_unit,
            attention_dropout_rate,
            causal,
            flashattention,
        )
        self.attn_ln = torch.nn.LayerNorm(att_unit)

        if cross_attention:
            self.cross_attn = MultiHeadedAttention(
                head,
                att_unit,
                attention_dropout_rate,
                False,
                flashattention,
            )
            self.cross_attn_ln = torch.nn.LayerNorm(att_unit)
        else:
            self.cross_attn = None
            self.cross_attn_ln = None

        self.ffn = PositionwiseFeedForward(att_unit, unit, dropout_rate)
        # self.ffn_ln = torch.nn.LayerNorm(att_unit)

    def forward(
        self,
        input: torch.Tensor,
        input_masks: torch.Tensor = None,
        src: torch.Tensor = None,
        src_masks: torch.Tensor = None,
        cache: Dict = None,
    ):
        ######## Jinchuan: Take care of the layer norms here. Order is wrong!!!
        # self-attn
        x = self.attn_ln(input)
        x = x + self.attn(x, x, x, input_masks, cache)

        # cross-attn
        if self.cross_attn and src is not None:
            x = self.cross_attn_ln(x)
            x = x + self.cross_attn(x, src, src, src_masks, cache)

        # feed-forward
        # x = self.ffn_ln(x)
        x = x + self.ffn(x)

        return x