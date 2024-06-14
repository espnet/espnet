from typing import Optional

from torch import Tensor, nn

from espnet2.speechlm.module.transformer import (
    ResidualAttentionBlock,
    TransformerDecoder,
)


class AdaLN(nn.Module):
    def __init__(self, n_state, eps=1e-5):
        super().__init__()
        self.weight = nn.Linear(n_state, n_state, bias=False)
        self.bias = nn.Linear(n_state, n_state, bias=False)
        nn.init.constant_(self.weight.weight, 1.0)
        nn.init.constant_(self.bias.weight, 0.0)

        self.n_state = n_state
        self.eps = eps

    def forward(self, x: Tensor, level_emb: Tensor):
        w = self.weight(level_emb).unsqueeze(1)
        b = self.bias(level_emb).unsqueeze(1)
        x = nn.functional.layer_norm(x, (self.n_state,), eps=self.eps)
        x = w * x + b
        return x


class ResidualAttentionBlockAdaLM(ResidualAttentionBlock):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super(ResidualAttentionBlockAdaLM, self).__init__(
            n_state=n_state,
            n_head=n_head,
            cross_attention=cross_attention,
        )

        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                setattr(self, name, AdaLN(n_state))

    def forward(
        self,
        x: Tensor,
        level: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x, level), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = (
                x
                + self.cross_attn(self.cross_attn_ln(x, level), xa, kv_cache=kv_cache)[
                    0
                ]
            )
        x = x + self.mlp(self.mlp_ln(x, level))
        return x


class ValleNARDecoder(TransformerDecoder):
    def __init__(
        self,
        n_level: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        causal: bool = True,
        layer_class=ResidualAttentionBlockAdaLM,
    ):
        super(ValleNARDecoder, self).__init__(
            n_ctx=n_ctx,
            n_state=n_state,
            n_head=n_head,
            n_layer=n_layer,
            causal=causal,
            layer_class=layer_class,
        )

        self.level_emb = nn.Embedding(n_level, n_state)
        self.ln = AdaLN(n_state)

    def forward(self, x: Tensor, level: Tensor, kv_cache: Optional[dict] = None):
        level = self.level_emb(level)

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = x + self.pos_emb.weight[offset : offset + x.shape[1]].unsqueeze(0)

        for block in self.blocks:
            x = block(x, level=level, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x, level)
        return x
