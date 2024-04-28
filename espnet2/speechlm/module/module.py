import torch
import math
# import version

from typing import Dict
from espnet2.speechlm.net_utils import causal_mask


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
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.dropout_rate = dropout_rate
        self.causal = causal
        self.flashattention = flashattention

        # check flashattention availability
        # torch_version = torch.__version__
        # if flashattention and version.parse(torch_version) < version.parse("2.0.1"):
        #     raise ValueError(f"Upgrade Pytorch 2.0.1+ to use flashattention")

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask, cache=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        if self.causal and mask is not None:
            raise ValueError("Cannot require causality when mask is provided.")

        if self.causal and not (query.size(1) == key.size(1) == value.size(1)):
            raise ValueError("causality is only for self-attention")

        q, k, v = self.forward_qkv(query, key, value)

        # (Jinchuan) Try always use flash-attention for better efficiency
        if not self.flashattention:
            if self.causal:
                mask = causal_mask(query.size(1), query.device)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask)

        else:
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask.bool().unsqueeze(1) if mask is not None else None,
                dropout_p=self.dropout_rate,
                is_causal=self.causal,
            )
            x = x.transpose(1, 2).flatten(2, 3)
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

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
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

    def forward(
        self,
        input: torch.Tensor,
        input_masks: torch.Tensor = None,
        src: torch.Tensor = None,
        src_masks: torch.Tensor = None,
        cache: Dict = None,
    ):
        # self-attn
        x = self.attn_ln(input)
        x = x + self.attn(x, x, x, input_masks, cache)

        # cross-attn
        if self.cross_attn and src is not None:
            x = self.cross_attn_ln(x)
            x = x + self.cross_attn(x, src, src, src_masks, cache)

        # feed-forward
        x = x + self.ffn(x)

        return x