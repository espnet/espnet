import torch

from espnet.nets.scorer_interface import ScorerInterface


class Qwen2HFScorer(ScorerInterface):
    def __init__(
        self,
        model,
        input_ids,
        attention_mask,
        input_features=None,
        feature_attention_mask=None,
    ):
        super().__init__()
        self.model = model.eval()
        self.prefill_input_ids = input_ids
        self.prefill_attn_mask = attention_mask
        self.prefill_feats = input_features
        self.prefill_feat_mask = feature_attention_mask

        # Ensure caching on
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

    def init_state(self, xs):
        return {
            "past_kv": None,
            "step": 0,
        }

    @torch.no_grad()
    def score(self, ys, state, xs):
        """Forward scoring function.

        ys: 1D LongTensor of generated tokens so far
        state: dict with 'past_kv' and 'step'
        """
        if state["past_kv"] is None:
            # PREFILL once with audio and full text prompt
            out = self.model(
                input_ids=self.prefill_input_ids,
                attention_mask=self.prefill_attn_mask,
                input_features=self.prefill_feats,
                feature_attention_mask=self.prefill_feat_mask,
                use_cache=True,
                return_dict=True,
            )
            logits = out.logits[:, -1]
            past_kv = out.past_key_values

            past_len = (
                past_kv[0][0].size(-2)
                if past_kv is not None
                else self.prefill_input_ids.size(1)
            )

            new_state = {"past_kv": past_kv, "step": int(past_len)}
            logp = torch.log_softmax(logits, dim=-1).squeeze(0)
            return logp, new_state

        # INCREMENTAL: feed ONLY last token; do not pass audio again
        last_tok = ys[-1].view(1, 1)  # shape (1, 1)

        # Compute position_ids explicitly from cache length to stop degeneration
        past_kv = state["past_kv"]
        past_len = state["step"]
        # Fallback to read from KV if step got desynced
        if isinstance(past_kv, tuple) and len(past_kv) > 0 and len(past_kv[0]) > 0:
            kv_len = past_kv[0][0].size(-2)
            if kv_len != past_len:
                past_len = kv_len

        position_ids = torch.tensor(
            [[past_len]], dtype=torch.long, device=last_tok.device
        )

        out = self.model(
            input_ids=last_tok,
            past_key_values=past_kv,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits[:, -1]
        new_past = out.past_key_values
        new_step = past_len + 1

        logp = torch.log_softmax(logits, dim=-1).squeeze(0)
        return logp, {"past_kv": new_past, "step": new_step}

    def select_state(self, states, idx):
        if hasattr(idx, "tolist"):
            idx = idx.tolist()
        out = []
        for i in idx:
            assert i < len(
                states
            ), f"Index {i} out of range, len(states): {len(states)}"
            s = states[i]
            out.append({"past_kv": s["past_kv"], "step": int(s["step"])})
        return out

    def final_score(self, states):
        return 0.0
